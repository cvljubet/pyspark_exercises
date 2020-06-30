# Databricks notebook source
# MAGIC %md # Spark DataFrames Project Exercise 

# COMMAND ----------

# MAGIC %md Let's get some quick practice with your new Spark DataFrame skills, you will be asked some basic questions about some stock market data, in this case Walmart Stock from the years 2012-2017. This exercise will just ask a bunch of questions, unlike the future machine learning exercises, which will be a little looser and be in the form of "Consulting Projects", but more on that later!
# MAGIC 
# MAGIC For now, just answer the questions and complete the tasks below.

# COMMAND ----------

# MAGIC %md #### Use the walmart_stock.csv file to Answer and complete the  tasks below!

# COMMAND ----------

# MAGIC %md #### Start a simple Spark Session

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Excercise').getOrCreate()

# COMMAND ----------

# MAGIC %md #### Load the Walmart Stock CSV File, have Spark infer the data types.

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/walmart_stock.csv', header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md #### What are the column names?

# COMMAND ----------

df.show()

# COMMAND ----------

df.head(1)

# COMMAND ----------

# MAGIC %md Column names: 

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md #### What does the Schema look like?

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md #### Print out the first 5 columns.

# COMMAND ----------

df.select(df.columns[:5]).show()

# COMMAND ----------

# MAGIC %md #### Use describe() to learn about the DataFrame.

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md ## Bonus Question!
# MAGIC #### There are too many decimal places for mean and stddev in the describe() dataframe. Format the numbers to just show up to two decimal places. Pay careful attention to the datatypes that .describe() returns, we didn't cover how to do this exact formatting, but we covered something very similar. [Check this link for a hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.cast)
# MAGIC 
# MAGIC If you get stuck on this, don't worry, just view the solutions.

# COMMAND ----------

summary = df.describe()

# COMMAND ----------

summary.dtypes

# COMMAND ----------

#El método describe() dejó todas las columnas como strings, asi que las convertiré a double

#nombre de las columnas
col = summary.columns


# COMMAND ----------

summ2 = summary   #hago una copia para trabajar sobre esa copia


# COMMAND ----------

for i in range(1,len(col)):  #desde 1 porque la 1ra columna debe quedar como string
  summ2 = summ2.withColumn(summ2.columns[i],summ2[i].cast('double'))

# COMMAND ----------

#summ2 = summ2.select(summ2['summary'],
#                    summ2['Open'].cast('double'),
#                    summ2['High'].cast('double'),
#                    summ2['Low'].cast('double'),
#                    summ2['Close'].cast('double'),
#                    summ2['Volume'].cast('double'),
#                    summ2['Adj Close'].cast('double'))

# COMMAND ----------

#print((summ2.count(), len(summ2.columns)))
summ2.printSchema()

# COMMAND ----------

#Ahora mostraremos las columnas de summ2 con 2 decimales
from pyspark.sql.functions import format_number

# COMMAND ----------

#Esta vez haré algo distinto al loop for anterior
summ2 = summ2.select(summ2['summary'],
                    format_number('Open',2).alias('Open'),
                    format_number('High',2).alias('High'),
                    format_number('Low',2).alias('Low'),
                    format_number('Close',2).alias('Close'),
                    format_number('Volume',2).alias('Volume'),
                    format_number('Adj Close',2).alias('Adj Close'))

# COMMAND ----------

# MAGIC %md #### Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day.

# COMMAND ----------

dfnew = df.withColumn('HV Ratio', df['High']/df['Volume'])
#Nos quedamos solo con la columna HV Ratio
dfnew = dfnew.select('HV Ratio')

# COMMAND ----------

dfnew.show()

# COMMAND ----------

# MAGIC %md #### What day had the Peak High in Price?

# COMMAND ----------

#Aplicaremos un filtro para obtener el día con el mayor precio
max_high = df.agg({'High':'max'}).collect()[0][0] #Así puedo obtener el valor máximo y guardarlo como un número (usando .collect()[0][0])
type(max_high)
print(max_high)
df.filter(df['High']==max_high).select('Date','High').show()


# COMMAND ----------

#Otra forma de obtener el máximo:
df.orderBy(df['High'].desc()).head(1)[0][0]
#Explicación de los índices:
#(1): fila 1, primer [0]: tomo el objeto fila, seguno [0]: tomo el 1er elemento de ese objeto, que sería la fecha

# COMMAND ----------

# MAGIC %md #### What is the mean of the Close column?

# COMMAND ----------

df.agg({'Close':'mean'}).show()

# COMMAND ----------

#Otra forma:
from pyspark.sql.functions import avg
df.select(avg('Close')).show()

# COMMAND ----------

# MAGIC %md #### What is the max and min of the Volume column?

# COMMAND ----------

from pyspark.sql.functions import max, min


# COMMAND ----------

df.select(max('Volume'),min('Volume')).show()

# COMMAND ----------

#Nota sobre guardar datos
data1 = df.select(max('Volume')) #Aquí estoy guardando un dataframe, puedo usar despues .show()
data2 = df.select(max('Volume')).collect() #Aquí estoy guardando una lista, para mostrar el valor 
                                          #puedo añadir luego [0][0]

# COMMAND ----------

data1.show()
data2[0][0]

# COMMAND ----------

# MAGIC %md #### How many days was the Close lower than 60 dollars?

# COMMAND ----------

df.filter('Close<60').count()
#O sin SQL
df.filter(df['Close']<60).count()

# COMMAND ----------

#También se puede usar count como función, usando select
from pyspark.sql.functions import count
filtered = df.filter(df['Close']<60)
filtered.select(count('Close')).show()

# COMMAND ----------

# MAGIC %md #### What percentage of the time was the High greater than 80 dollars ?
# MAGIC #### In other words, (Number of Days High>80)/(Total Days in the dataset)

# COMMAND ----------

df.filter('High>80').count()/df.count()*100

# COMMAND ----------

# MAGIC %md #### What is the Pearson correlation between High and Volume?
# MAGIC #### [Hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameStatFunctions.corr)

# COMMAND ----------

from pyspark.sql.functions import corr

# COMMAND ----------


corr = df.corr('High','Volume')

# COMMAND ----------

corr

# COMMAND ----------

# MAGIC %md #### What is the max High per year?

# COMMAND ----------

from pyspark.sql.functions import year

# COMMAND ----------

#Creamos una columna con el año
df = df.withColumn('Year', year(df['Date']))

#Agrupamos los datos por año y sacamos el máximo
df.groupBy('Year').max().select('Year','max(Year)').show()

# COMMAND ----------

# MAGIC %md #### What is the average Close for each Calendar Month?
# MAGIC #### In other words, across all the years, what is the average Close price for Jan,Feb, Mar, etc... Your result will have a value for each of these months. 

# COMMAND ----------

from pyspark.sql.functions import month

# COMMAND ----------

#Añado columna con mes
df = df.withColumn('Month',month(df['Date']))

#Agrupamos por mes y calculamos el promedio
df_month = df.groupBy('Month').mean()

#Seleccionamos las columnas deseadas
df_month.select('Month','avg(Close)').orderBy('month').show()

# COMMAND ----------

# MAGIC %md # Great Job!
