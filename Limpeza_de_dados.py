# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Projeto Final - Limpeza de Dados

# %% [markdown]
# ## Setup

# %%
from pyspark.sql import functions as sf
import pyspark.sql.functions as f

# Modeling
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# %%
## Bibliotecas Gráficas
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Criar a sessao do Spark
from pyspark.sql import SparkSession
spark = SparkSession \
            .builder \
            .master("local[4]") \
            .appName("nyc_caio_garcia") \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-azure:3.3.4,com.microsoft.azure:azure-storage:8.6.6") \
            .getOrCreate()

# %%
# Acesso aos dados na nuvem
STORAGE_ACCOUNT = 'dlspadseastusprod'
CONTAINER = 'big-data-comp-nuvem'
FOLDER = 'airline-delay'
TOKEN = 'lSuH4ZI9BhOFEhCF/7ZQbrpPBIhgtLcPDfXjJ8lMxQZjaADW4p6tcmiZGDX9u05o7FqSE2t9d2RD+ASt0YFG8g=='

spark.conf.set("fs.azure.account.key." + STORAGE_ACCOUNT + ".blob.core.windows.net", TOKEN)

# %% [markdown]
# ### Schema
# Schema definido de acordo com o dicionário de dados em `projeto_final_dicionário.xlsx`

# %%
from pyspark.sql.types import *

labels = (('FL_DATE', TimestampType()),
          ('OP_CARRIER', StringType()),
          ('OP_CARRIER_FL_NUM', IntegerType()),
          ('ORIGIN', StringType()),
          ('DEST', StringType()),
          ('CRS_DEP_TIME', IntegerType()),
          ('DEP_TIME', FloatType()),
          ('DEP_DELAY', FloatType()),
          ('TAXI_OUT', FloatType()),
          ('WHEELS_OFF', FloatType()),
          ('WHEELS_ON', FloatType()),
          ('TAXI_IN', FloatType()),
          ('CRS_ARR_TIME', IntegerType()),
          ('ARR_TIME', FloatType()),
          ('ARR_DELAY', FloatType()),
          ('CANCELLED', FloatType()),
          ('CANCELLATION_CODE', StringType()),
          ('DIVERTED', FloatType()),
          ('CRS_ELAPSED_TIME', FloatType()),
          ('ACTUAL_ELAPSED_TIME', FloatType()),
          ('AIR_TIME', FloatType()),
          ('DISTANCE', FloatType()),
          ('CARRIER_DELAY', FloatType()),
          ('WEATHER_DELAY', FloatType()),
          ('NAS_DELAY', FloatType()),
          ('SECURITY_DELAY', FloatType()),
          ('LATE_AIRCRAFT_DELAY', StringType()))

schema = StructType([StructField(x[0], x[1], True) for x in labels])

# %%
# Columns with values in minutes
minute_columns = ["TAXI_OUT","TAXI_IN","DEP_DELAY","ARR_DELAY","AIR_TIME","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME",
                  "CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"]

# Subset from minute columns with no data leak from moment of take off
clean_min_columns = ["TAXI_OUT", "DEP_DELAY", "CRS_ELAPSED_TIME"]

# Columns with time information on format 'hhmm'
# Not proper for numerical manipulation
odd_format_columns = ["CRS_DEP_TIME","DEP_TIME","WHEELS_OFF","WHEELS_ON","ARR_TIME","CRS_ARR_TIME"]

# %% [markdown]
# ### Carregamento de dados
# Dados carregados da nuvem como spark data frame

# %% [markdown]
# Exemplo para o ano de 2009:

# %%
config = spark.sparkContext._jsc.hadoopConfiguration()
config.set("fs.azure.account.key." + STORAGE_ACCOUNT + ".blob.core.windows.net", TOKEN)
sc = spark.sparkContext

df_exemple = spark.read.csv("wasbs://{}@{}.blob.core.windows.net/{}/2009.csv"\
                    .format(CONTAINER, STORAGE_ACCOUNT, FOLDER), header=True, schema=schema)
df_exemple.take(2)

# %% [markdown]
# Como temos dados para os anos de 2009 até o ano de 2018. Iremos criar um dicionario contendo os dataframes de cada ano separadamente. 

# %%
# Criando dicionario de dataframes
df_for_year = {}

# Loop lendo arquivo de cada ano e salvando no dicionario
for year in range(2009, 2019):
    # Ajustando o caminho
    file_path = "wasbs://{}@{}.blob.core.windows.net/{}/{}.csv"\
                    .format(CONTAINER, STORAGE_ACCOUNT, FOLDER, year)
    
    # lendo arquivo csv 
    df_name = "df_{}".format(year)
    df = spark.read.csv(file_path, header=True, schema=schema)
    
    # Adicionando df ao dicionario de dataframes 
    df_for_year[df_name] = df
    

# Visualizando as primeiras linhas de 2012
df_for_year["df_2012"].take(5)

# %% [markdown]
# Com todos os dataframes pré-importados(lazy) podemos realizar um merge unindo todos os anos.

# %%
# Importando função reduce para realizar o merge
from functools import reduce 

# %%
# União de todos os DataFrames em um único Data frame
df_final = reduce(sf.DataFrame.union, df_for_year.values())

# Criando a coluna "year" baseada na coluna "date"
df_final = df_final.withColumn("year", sf.year("FL_DATE"))

# Exibindo as primeiras linhas
df_final.take(10)

# %% [markdown]
# Assim, temos o dataframe com todos os anos.

# %%
OBSERVACOES = df_final.count()
assert (OBSERVACOES == 61556964)

# %%
OBSERVACOES

# %% [markdown]
# A base contem 61556964 observações.

# %%
CANCELAMENTOS = df_final.filter(df_final.CANCELLED == 1).count()
assert (CANCELAMENTOS == 973209)

# %%
CANCELAMENTOS

# %% [markdown]
# Dos voos na base, 973209 foram cancelados.

# %% [markdown]
# ## Tratamento de dados faltantes

# %% [markdown]
# Das 27 colunas na base de dados, 16 tem valores faltantes. Os dados ausentes na coluna CANCELATION_CODE sao 100% consistentes com a informacao de cancelamento, isto eh, apenas os voos que nao foram cancelados tem a coluna CANCELATION_CODE vazia. As outras 15 colunas podem ser agrupadas em 3 grupos: _Voo_, _Chegada_ e _Atrasos_.

# %% [markdown]
# ### Resumo

# %%
missing_counts = df_final.select([sf.col(column).isNull().cast("int").alias(column) for column in df_final.columns]) \
                       .groupBy() \
                       .sum()

# %%
# Criando dataframe de colunas com valores zerados
missing_counts_df = missing_counts.toPandas().transpose()

# Filtrando apenas colunas com valores nulos
missing_counts_df = missing_counts_df[missing_counts_df[0]>0]

# Renomeando a coluna
missing_counts_df = missing_counts_df.rename(columns={0:"nulos"})

# %%
# Contando número de colunas com valores nulos
print("Número de colunas com valores faltantes:")
missing_counts_df.count()[0]

# %%
# Cálculando porcentagem de valores faltantes
missing_counts_df["%nulos"] = (missing_counts_df["nulos"]/OBSERVACOES) * 100

# Ordenando por % de nulos
missing_counts_df = missing_counts_df.sort_values("%nulos", ascending=False)

# Visualizando resultados
missing_counts_df

# %%
# Visualizando missing em gráfico de barras
# Ajustando o tamanho da figura
plt.figure(figsize=(10, 4))

# Plotando o gráfico de barras
sns.barplot(x=missing_counts_df.index, y=missing_counts_df["%nulos"], color="red")

# Adicionando inclinação aos valores do eixo x
plt.xticks(rotation=45, ha='right')

# Adicionando título e rótulos aos eixos
plt.title('Gráfico de Barras')
plt.xlabel("Colunas")
plt.ylabel("%Nulos")

# Exibindo o gráfico
plt.show()

# %% [markdown]
# Conforme evidenciado no gráfico apresentado, nota-se que a coluna "CANCELLATION_CODE" exibe uma lacuna em praticamente 100% dos dados, enquanto as colunas "LATE_AIRCRAFT_DELAY", "NAS_DELAY", "WEATHER_DELAY", "CARRIER_DELAY" e "SECURITY_DELAY" apresentam uma ausência de informações em torno de 80%.

# %% [markdown]
# ### Cancelamentos

# %%
assert (df_final.filter((df_final.CANCELLATION_CODE.isNull()) &
                  (df_final.CANCELLED == 0)).count() ==
        df_final.filter(df_final.CANCELLED == 0).count())

# %% [markdown]
# Todos os valores faltantes de CANCELLATION_CODE são referentes a voos que não foram cancelados.

# %% [markdown]
# ### Voo

# %% [markdown]
# #### Testes

# %% [markdown]
# `DEP_TIME` e `DEP_DELAY`: co-ausentes, todos cancelados

# %%
assert (df_final.filter((df_final.DEP_TIME.isNull())   &
                  (df_final.DEP_DELAY.isNull())  &
                  (df_final.TAXI_OUT.isNull())   &
                  (df_final.WHEELS_OFF.isNull()) &
                  (df_final.WHEELS_ON.isNull())  &
                  (df_final.TAXI_IN.isNull())    &
                  (df_final.ARR_TIME.isNull())).count() ==
        df_final.filter(df_final.DEP_TIME.isNull()).count())

assert (df_final.filter((df_final.DEP_TIME.isNull())   &
                  (df_final.DEP_DELAY.isNull())).count() ==
        df_final.filter(df_final.DEP_TIME.isNull()).count())

# %%
assert (df_final.filter((df_final.DEP_TIME.isNull())   &
                  (df_final.CANCELLED == 1)).count() ==
        df_final.filter(df_final.DEP_TIME.isNull()).count())

# %% [markdown]
# `TAXI_OUT` e `WHEELS_OFF`: co-ausentes e cancelados

# %%
assert (df_final.filter((df_final.TAXI_OUT.isNull())   &
                  (df_final.WHEELS_OFF.isNull()) &
                  (df_final.WHEELS_ON.isNull())  &
                  (df_final.TAXI_IN.isNull())    &
                  (df_final.ARR_TIME.isNull())).count() ==
        df_final.filter(df_final.TAXI_OUT.isNull()).count())

assert (df_final.filter((df_final.TAXI_OUT.isNull())   &
                  (df_final.WHEELS_OFF.isNull())).count() ==
        df_final.filter(df_final.TAXI_OUT.isNull()).count())

# %%
assert (df_final.filter((df_final.TAXI_OUT.isNull())   &
                  (df_final.CANCELLED == 1)).count() ==
       df_final.filter(df_final.TAXI_OUT.isNull()).count())

# %% [markdown]
# `WHEELS_ON`, `TAXI_IN` e `ARR_TIME`: co-ausentes

# %%
assert (df_final.filter((df_final.WHEELS_ON.isNull())  &
                  (df_final.TAXI_IN.isNull())    &
                  (df_final.ARR_TIME.isNull())).count() ==
        df_final.filter(df_final.TAXI_IN.isNull()).count())

# %%
assert (df_final.filter((df_final.TAXI_IN.isNull())   &
                  (df_final.CANCELLED == 1)).count() ==
        df_final.filter(df_final.CANCELLED == 1).count())

# %%
assert (df.filter((df.TAXI_IN.isNull()) &
                  (df.CANCELLED == 0)   &
                  (df.DIVERTED == 1)).count() ==
        df.filter((df.TAXI_IN.isNull()) &
                  (df.CANCELLED == 0)).count())

# %% [markdown]
# #### Análise

# %% [markdown]
# O grupo _Voo_ apresenta uma relação entre voos cancelados e as 7 variáveis:
#  - DEP_TIME
#  - DEP_DELAY
#  - TAXI_OUT
#  - WHEELS_OFF
#  - WHEELS_ON
#  - TAXI_IN
#  - ARR_TIME
#  
# Os valores faltantes para WHEELS_ON, TAXI_IN e ARR_TIME coincidem nas mesmas observações (com uma exceção descrita mais abaixo). Todos os voos cancelados se encontram dentre essas observações. Os valores faltantes para TAXI_OUT e WHEELS_OFF coincidem nas mesmas observações, todas referentes a voos cancelados. Finalmente, os valores faltantes de DEP_TIME e DEP_DELAY coincidem nas mesmas observações, todas com valores faltantes para TAXI_OUT.
#
# Todos os voos que não foram cancelados mas não tem informação da hora de aterrisagem (`(df.WHEELS_ON.isNull()) & (df.CANCELLED == 0)`) foram redirecionados para um aeroporto diferente do aeroporto destino original (`df.DIVERTED == 1`)
#
# Destas relações, supomos:
#  - A diferença entre DEP_TIME e WHEELS_OFF pode ser devido a voos que chegam a sair do chão antes de serem cancelados, e voos que são cancelados após o embarque mas antes da decolagem.
#  - Nenhum desses valores faltantes parece implausível o suficiente para assumirmos erro nos dados baseado apenas nessa análise. Alguns desses dados podem vir a ser retirados mesmo assim por questão de propriedades dos algorítmos utilizados mais a frente.

# %% [markdown]
# ### Chegada

# %% [markdown]
# ### Atrasos
# TODO

# %% [markdown]
# #### Testes

# %%
assert (df.filter((df.CANCELLED == 0) &
                  (df.DEP_DELAY > 0)).count() == 
        2252608)

# %%
assert (df.filter((df.CANCELLED == 0) &
                  (df.ARR_DELAY > 0)).count() ==
        2402990)

# %%
assert (df.filter((df.CANCELLED == 0) &
                  ((df.DEP_DELAY > 0) |
                   (df.ARR_DELAY > 0))).count() ==
        3052688)

# %%
assert (OBSERVACOES - df.filter(df.CARRIER_DELAY.isNull()).count() ==
        1170501)

# %%
assert (df.filter((df.CARRIER_DELAY.isNull()) &
                  (df.WEATHER_DELAY.isNull()) &
                  (df.NAS_DELAY.isNull()) &
                  (df.SECURITY_DELAY.isNull()) &
                  (df.LATE_AIRCRAFT_DELAY.isNull())).count() ==
        df.filter(df.CARRIER_DELAY.isNull()).count())

# %%
df.filter(df.CARRIER_DELAY == 0).select(df.OP_CARRIER_FL_NUM, df.CARRIER_DELAY).take(10)

# %% [markdown]
# #### Análise

# %% [markdown]
# Todos os dados faltantes referentes a categoria de atraso coincidem nas mesmas observações.
#
# Há menos observações com informação sobre a causa do atraso do que voos atrasados, independente se medindo o atraso de saída ou de chegada.

# %% [markdown]
# ### Anomalia

# %%
assert (df.filter((df.ACTUAL_ELAPSED_TIME.isNull()) &
                  (df.AIR_TIME == 0)                &
                  (df.WHEELS_ON.isNull())           &
                  (df.ARR_TIME.isNull())            &
                  (df.ARR_DELAY == 0)               &
                  (df.TAXI_IN == 0)                 &
                  (df.CANCELLED == 0)).count() == 1)

# %% [markdown]
# Uma mesma observação é responsavel pela discrepância na quantidade total de valores faltantes entre WHEELS_ON, TAXI_IN e ARR_TIME, e ARR_DELAY, ACTUAL_ELAPSED_TIME e AIR_TIME.
# Um valor de `AIR_TIME == 0` nao faz sentido para um voo que não foi cancelado, e o mesmo se aplica a `TAXI_IN == 0`. Ao retirar essa observação da base, a análise de dados faltantes por grupo torna-se mais consistente.

# %% [markdown]
# ## Consistencia

# %%
assert (df.filter(df.AIR_TIME + df.TAXI_IN + df.TAXI_OUT != df.ACTUAL_ELAPSED_TIME).count() == 0)

# %%
assert (df.filter((df.CANCELLED == 1) &
                  (df.DIVERTED == 1)).count() == 0)

# %%
assert (df.filter((df.DEP_TIME % 1 != 0) | (df.DEP_DELAY % 1 != 0)).count() == 0)

# %% [markdown]
# # Modelagem

# %%
# This list includes all values not known at the moment of takeoff
# except `ARR_DELAY` which will be used as target variable
take_off_leak = ["WHEELS_ON","TAXI_IN","ARR_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME",
                 "CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY"]

# %%
take_off_df = df_final.drop(*take_off_leak)\
                      .filter(df_final.CANCELLED == 0)\
                      .filter(df_final.DIVERTED == 0)\
                      .filter(df_final.CRS_ARR_TIME.isNotNull())\
                      .filter(df_final.ARR_DELAY.isNotNull())\
                      .filter(df_final.DEP_DELAY.isNotNull())

# %%
missing_counts = take_off_df.select([sf.col(column).isNull().cast("int").alias(column) for column in take_off_df.columns]) \
                       .groupBy() \
                       .sum()

# %%
# Criando dataframe de colunas com valores zerados
missing_counts_df = missing_counts.toPandas().transpose()

# Filtrando apenas colunas com valores nulos
missing_counts_df = missing_counts_df[missing_counts_df[0]>0]

# Renomeando a coluna
missing_counts_df = missing_counts_df.rename(columns={0:"nulos"})

# %%
# Cálculando porcentagem de valores faltantes
missing_counts_df["%nulos"] = (missing_counts_df["nulos"]/OBSERVACOES) * 100

# Ordenando por % de nulos
missing_counts_df = missing_counts_df.sort_values("%nulos", ascending=False)

# Visualizando resultados
missing_counts_df

# %% [markdown]
# ## Train/Test Split

# %%
train_df, test_df = take_off_df.randomSplit([0.8,0.2], seed=42)
toy_df = train_df.sample(False, 0.01, seed=42)

# %%
print("Train set count:", train_df.count())
print("Test set count:", test_df.count())
print("Toy set count:", toy_df.count())

# %% [markdown]
# ## Feature Engineering: One-Hot-Enconding

# %%
cat_features = ["OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST"]

indexOutputCols = [x + 'Index' for x in cat_features]

oheOutputCols = [x + 'OHE' for x in cat_features]

stringIndex = StringIndexer(inputCols = cat_features,
                            outputCols = indexOutputCols,
                            handleInvalid = 'skip')

oheEncoder = OneHotEncoder(inputCols = indexOutputCols,
                           outputCols = oheOutputCols)

# %%
num_features = ["TAXI_OUT", "DEP_DELAY", "CRS_ELAPSED_TIME", "DISTANCE"]

numVecAssembler = VectorAssembler(inputCols = num_features,
                                  outputCol = 'features',
                                  handleInvalid = 'skip')

stdScaler = StandardScaler(inputCol = 'features',
                           outputCol = 'features_scaled')

# %% [markdown]
# ## Assembling dos vetores

# %%
assembleInputs = oheOutputCols + ['features_scaled']

vecAssembler = VectorAssembler(inputCols = assembleInputs,
                               outputCol = 'features_vector')

# %%
stages = [stringIndex, oheEncoder, numVecAssembler, stdScaler, vecAssembler]

# %% [markdown]
# ## Criação do Pipeline

# %%
# Criacao do pipeline de transformacao
transform_pipeline = Pipeline(stages=stages)

# Aplicacao do pipeline nos dados de treino
fitted_transformer = transform_pipeline.fit(train_df)
transformed_train_df = fitted_transformer.transform(train_df)

transformed_train_df.limit(10).toPandas()

# %% [markdown]
# ## Model Training

# %%
model = LinearRegression(maxIter = 5, # pode causar overfitting
                         solver = 'auto',
                         labelCol = 'ARR_DELAY',
                         featuresCol = 'features_vector',
                         elasticNetParam = 0.2,
                         regParam = 0.02)

pipe_stages = stages + [model]

pipe = Pipeline(stages=pipe_stages)

# %%
fitted_pipe = pipe.fit(toy_df)

# %% [markdown]
# ## Model performance evaluation

# %%
preds = fitted_pipe.transform(test_df)

# %%
preds.limit(10).toPandas()

# %%
rmse = RegressionEvaluator(labelCol = 'ARR_DELAY',
                           metricName = 'rmse').evaluate(preds)

# %%
print("RMSEof Prediction on test set:", rmse)

# %%
results = []
results.append({"run": 1,
                "rmse": rmse,
                "model": "LinearRegression",
                "params": "maxIter = 5, solver = 'auto', labelCol = 'ARR_DELAY', featuresCol = 'features_vector', elasticNetParam = 0.2, regParam = 0.02"})

# %%
df_final.drop(*take_off_leak)\
                      .filter(df_final.CANCELLED == 0)\
                      .filter(df_final.DIVERTED == 0)\
                      .filter(df_final.CRS_ARR_TIME.isNotNull())\
                      .filter(df_final.ARR_DELAY.isNotNull())\
                      .filter(df_final.DEP_DELAY.isNotNull()).count()
