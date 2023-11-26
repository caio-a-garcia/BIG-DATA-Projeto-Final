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
          ('OP_CARRRIER', StringType()),
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

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Cancelamentos

# %%
assert (df.filter((df.CANCELLATION_CODE.isNull()) &
                  (df.CANCELLED == 0)).count() ==
        df.filter(df.CANCELLED == 0).count())

# %% [markdown]
# Todos os valores faltantes de CANCELLATION_CODE são referentes a voos que não foram cancelados.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Voo

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# #### Testes

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

# %%
assert (df.filter((df.WHEELS_ON.isNull())  &
                  (df.TAXI_IN.isNull())    &
                  (df.ARR_TIME.isNull())).count() ==
        df.filter(df.TAXI_IN.isNull()).count())

# %%
assert (df.filter((df.TAXI_IN.isNull())   &
                  (df.CANCELLED == 1)).count() ==
        df.filter(df.CANCELLED == 1).count())

# %% [markdown] jp-MarkdownHeadingCollapsed=true
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
# Destas relações, supomos:
#  - Alguns voos parecem não ter sua chegada propriamente registrada. Casos de pousos de emergencia em localização diferente da planejada são plausíveis mas supomos que sejam menos frequentes. Esses casos (`(df.WHEELS_ON.isNull()) & (df.CANCELLED == 0)`) podem vir a ser uma categoria relevante na análise.
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
assert (df.filter((df.CARRIER_DELAY.isNull()) &
                  ((df.DEP_DELAY > 0) |
                   (df.ARR_DELAY > 0))).count() ==
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
