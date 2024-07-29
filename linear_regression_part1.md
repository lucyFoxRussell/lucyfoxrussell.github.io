# House Price Prediction using Linear Regression: Part 1


## Introduction of project
Goal: My goal for this project is to use the square foot and price of houses to predict the price of others houses when given the sqft. 

Part 1: To achieve my goal I will start by using the data of two houses to produce a very simple linear regression model. This will help me to understand the fundamentals of Linear Regression. 

To keep this project very simple as an initial starting point into understanding Linear Regression I will just use data for two houses.  
House 1: is a house with 1000 square feet(sqft) sold for £300,000.
House 2: is a house with 2000sqft sold for £500,000. 

Step 1: Create and format data<br>
Step 2: Calculate the equation of the line of best fit between the two points.<br>
Step 3: Plot the line<br>
Step 4: Use model to predict a house price when given a sqft.<br>

## Part 1: Linear Regression with two pieces of data

### Step 1: Create and format data


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# size_in_sqf_arr is the input variable (size in 1000 square feet) and price_in_pounds is the target.
size_in_sqft_arr = np.array([1.0, 2.0])
price_in_pounds_arr = np.array([300.0, 500.0])
```

### initial plot of data


```python
plt.scatter(size_in_sqft_arr, price_in_pounds_arr, marker='x')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of Pounds)')
plt.xlabel('Size (1000 sqft)')
plt.grid(True)
plt.show()
```


    
![png](output_6_0.png)
    


### Step 2: Calculate the equation of the line

My goal for this simple linear regressions is given the square foot and price of two houses work out the equation of the line of best fit which can then be used to figure out the house of any price given its square feet. I can do this by calculating the equation of the line using the formula below. <br>

<figure>
 <img src="https://thirdspacelearning.com/wp-content/uploads/2021/08/ymxc-What-is.png"   style="width:500px;height:250px;">
</figure>

#### Coordinates from data

We can use (x$^{(i)}$, y$^{(i)}$) to denote the $i^{th}$ training example. Since Python is zero indexed, (x$^{(0)}$, y$^{(0)}$) is (1.0, 300.0) and (x$^{(1)}$, y$^{(1)}$) is (2.0, 500.0). 


```python
# When i = 0 we get the values of data from the first house. 
# When i = 1 we get the values of data from the second house. 
i = 0
x_i = size_in_sqft_arr[i]
y_i = price_in_pounds_arr[i]
print(f"Coordinates at index 0 = ({x_i}, {y_i})")

i = 1
x_i = size_in_sqft_arr[i]
y_i = price_in_pounds_arr[i]
print(f"Coordinates at index 1 = ({x_i}, {y_i})")
```

    Coordinates at index 0 = (1.0, 300.0)
    Coordinates at index 1 = (2.0, 500.0)


#### Calculate Slope: m

<figure>
 <img src="https://www.inchcalculator.com/wp-content/uploads/2019/09/slope-equation.png"   style="width:300px;height:200px;">
</figure>

In order to calculate the slope/ coefficent of a line we can use the formula above. 


```python
x1, x2, y1, y2 = 1, 2, 300, 500
m = (y2-y1)/ (x2-x1)
print("m: ", m)
```

    m:  200.0


So the equation of this line is as follows y = 200(x) + b

#### Calculating Y intersect: b

Using the coordinates from the first house data we can plug x position into the equation 300 = 200(1) + b <br>
To double check this we can plug in the coordinate values from the second house. 500 = 200(2) + b. Which also evaluates to b = 100


```python
b = 300 - 200
print("b: ", b)
```

    b:  100


I have now calculated that m = 200 and b = 100. 
Therefore the equation of the line is y = 200(m) + 100

### Step 3: Plot the equation of the line Y = 200(x) + 100

Using the matplotlib library I can plot the equation of the line and the coordinates of data of the two houses on a chart. 


```python
x_values, y_values = np.linspace(0, 3, 100), 200 * x_values + 100
plt.plot(x_values, y_values, label='Y = 200x + 100')
plt.scatter(size_in_sqft_arr, price_in_pounds_arr, color='red', label='Known House Data Points', zorder=5, marker='x', s=100)
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of Pounds)')
plt.title('Housing Prices')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_18_0.png)
    


### Step 4: Use model to predict a house price.
The goal of this project was to be able to predict the house price of a house with a given sqft. <br>
The sqft can be edited below to return the predicted house price and will be marked in green on the plot below. 


```python
sqft = 1.25
predicted_price = (200 * sqft) + 100
print(f"The predicted value of a house with {sqft * 1000} square feet is £{predicted_price * 1000}")
```

    The predicted value of a house with 1250.0 square feet is £350000.0



```python
x_values, y_values = np.linspace(0, 3, 100), 200 * x_values + 100
plt.plot(x_values, y_values, label='Y = 200x + 100')
plt.scatter(size_in_sqft_arr, price_in_pounds_arr, color='red', label='Known House Data Points', zorder=5, marker='x', s=100)
plt.scatter(sqft, predicted_price, color='green', label='Prediction', zorder=5, marker='x', s=100)
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of Pounds)')
plt.title('Housing Prices')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_21_0.png)
    


### Part 1 Conclusion

I have managed to use the data of two house prices and corresponding sqft to calculate a line of best fit and then use this to make predictions of other house prices with a different sqft. In this regard this task has been very successful. However, only using this small sample size of data is very unreliable. In part 2 of this project I will use a greater sample size, splitting it into training and test data. This will allow me to gain a sense of accuracy of the model. 
