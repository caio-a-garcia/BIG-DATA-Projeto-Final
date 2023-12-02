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

# %%
config = spark.sparkContext._jsc.hadoopConfiguration()
config.set("fs.azure.account.key." + STORAGE_ACCOUNT + ".blob.core.windows.net", TOKEN)
sc = spark.sparkContext

df = spark.read.csv("wasbs://{}@{}.blob.core.windows.net/{}/2009.csv"\
                    .format(CONTAINER, STORAGE_ACCOUNT, FOLDER), header=True, schema=schema)
df.take(2)

# %%
OBSERVACOES = df.count()
assert (OBSERVACOES == 6429338)

# %% [markdown]
# A base contem 6429338 observações.

# %%
CANCELAMENTOS = df.filter(df.CANCELLED == 1).count()
assert (CANCELAMENTOS == 87038)

# %% [markdown]
# Dos voos na base, 87038 foram cancelados.

# %% [markdown]
# ## Tratamento de dados faltantes

# %% [markdown]
# Das 27 colunas na base de dados, 16 tem valores faltantes. Os dados ausentes na coluna CANCELATION_CODE sao 100% consistentes com a informacao de cancelamento, isto eh, apenas os voos que nao foram cancelados tem a coluna CANCELATION_CODE vazia. As outras 15 colunas podem ser agrupadas em 3 grupos: _Voo_, _Chegada_ e _Atrasos_.

# %% [markdown]
# ### Resumo

# %%
missing_counts = df.select([sf.col(column).isNull().cast("int").alias(column) for column in df.columns]) \
                       .groupBy() \
                       .sum()

# %%
missing_counts.toPandas().transpose()

# %% [markdown]
# ### Cancelamentos

# %%
assert (df.filter((df.CANCELLATION_CODE.isNull()) &
                  (df.CANCELLED == 0)).count() ==
        df.filter(df.CANCELLED == 0).count())

# %% [markdown]
# Todos os valores faltantes de CANCELLATION_CODE são referentes a voos que não foram cancelados.

# %% [markdown]
# ### Voo

# %% [markdown]
# #### Testes

# %% [markdown]
# `DEP_TIME` e `DEP_DELAY`: co-ausentes, todos cancelados

# %%
assert (df.filter((df.DEP_TIME.isNull())   &
                  (df.DEP_DELAY.isNull())  &
                  (df.TAXI_OUT.isNull())   &
                  (df.WHEELS_OFF.isNull()) &
                  (df.WHEELS_ON.isNull())  &
                  (df.TAXI_IN.isNull())    &
                  (df.ARR_TIME.isNull())).count() ==
        df.filter(df.DEP_TIME.isNull()).count())

assert (df.filter((df.DEP_TIME.isNull())   &
                  (df.DEP_DELAY.isNull())).count() ==
        df.filter(df.DEP_TIME.isNull()).count())

# %%
assert (df.filter((df.DEP_TIME.isNull())   &
                  (df.CANCELLED == 1)).count() ==
        df.filter(df.DEP_TIME.isNull()).count())

# %% [markdown]
# `TAXI_OUT` e `WHEELS_OFF`: co-ausentes e cancelados

# %%
assert (df.filter((df.TAXI_OUT.isNull())   &
                  (df.WHEELS_OFF.isNull()) &
                  (df.WHEELS_ON.isNull())  &
                  (df.TAXI_IN.isNull())    &
                  (df.ARR_TIME.isNull())).count() ==
        df.filter(df.TAXI_OUT.isNull()).count())

assert (df.filter((df.TAXI_OUT.isNull())   &
                  (df.WHEELS_OFF.isNull())).count() ==
        df.filter(df.TAXI_OUT.isNull()).count())

# %%
assert (df.filter((df.TAXI_OUT.isNull())   &
                  (df.CANCELLED == 1)).count() ==
        df.filter(df.TAXI_OUT.isNull()).count())

# %% [markdown]
# `WHEELS_ON`, `TAXI_IN` e `ARR_TIME`: co-ausentes

# %%
assert (df.filter((df.WHEELS_ON.isNull())  &
                  (df.TAXI_IN.isNull())    &
                  (df.ARR_TIME.isNull())).count() ==
        df.filter(df.TAXI_IN.isNull()).count())

# %%
assert (df.filter((df.TAXI_IN.isNull())   &
                  (df.CANCELLED == 1)).count() ==
        df.filter(df.CANCELLED == 1).count())

# %%
assert (df.filter((df.TAXI_IN.isNull()) &
                  (df.CANCELLED == 0)   &
                  (df.DIVERTED == 1)).count() ==
        df.filter((df.TAXI_IN.isNull()) &
                  (df.CANCELLED == 0)).count())

# %%
assert (df.filter((df.TAXI_IN.isNull()) &
                  (df.CANCELLED == 0)   &
                  (df.DIVERTED == 1)).count() !=
        df.filter(df.DIVERTED == 1).count())

# %%
assert (df.filter((df.TAXI_IN.isNotNull()) &
                  (df.DIVERTED == 1)).count() == 13040)

# %%
assert (df.filter((df.TAXI_IN.isNull()) &
                  (df.DIVERTED == 1)).count() == 2283)

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
# assert (df.filter((df.CARRIER_DELAY.isNull()) &
#                   ((df.DEP_DELAY > 0) |
#                    (df.ARR_DELAY > 0))).count() ==
#         df.filter(df.CARRIER_DELAY.isNull()).count())

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
take_off_df = df.drop(*take_off_leak)

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
cat_features = ["OP_CARRIER", "ORIGIN", "DEST"] # "OP_CARRIER_FL_NUM",

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
model = LinearRegression(maxIter = 25, # pode causar overfitting
                         solver = 'normal',
                         labelCol = 'ARR_DELAY',
                         featuresCol = 'features_vector',
                         elasticNetParam = 0.2,
                         regParam = 0.02)

pipe_stages = stages + [model]

pipe = Pipeline(stages=pipe_stages)

# %%
fitted_pipe = pipe.fit(train_df)

# %% [markdown]
# ## Model performance evaluation

# %%
preds = fitted_pipe.transform(test_df)

# %%
preds.limit(10).toPandas()
