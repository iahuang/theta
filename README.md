# theta
A time complexity analysis library written in Python. Supports multiple variables and arbitrary time complexity functions.

## Example
```python
import theta
import random

# Here is a test function that finds the sum of two lists.
def test_function(x: list[int], y: list[int]):
    lsum = 0

    for a in x:
        for b in y:
            lsum += a+b

    return lsum

# If we have N be the length of x and M be the length of y
N = theta.InputSizeVariable("n")
M = theta.InputSizeVariable("m")
# We can intuitively see that test_function is O(n*m).

# Create a sample data generator with various lengths of x,y and thus various values of N,M.
input_generator = (
    theta.FunctionInput(
        args=[[random.randint(0, 10) for _ in range(i1)], [random.randint(0, 10) for _ in range(i2)]],
        input_sizes={
            N: i1,
            M: i2
        }
    )
    for i1 in [5, 10, 15, 20, 30]
    for i2 in [5, 10, 15, 20, 30]
)

# Benchmark our data
data = theta.compile_runtime_data(
    f=test_function,
    function_inputs=input_generator,
    min_iters=1000,
)

# Print correlation values (higher == more correlated).
print("O(n)     ", theta.bigO_correlation(data, N)) # You can construct functions with N and M
                                                    # like you would like any other variable
print("O(m)     ", theta.bigO_correlation(data, M))
print("O(nm)    ", theta.bigO_correlation(data, N*M))
print("O(nlogm) ", theta.bigO_correlation(data, N*theta.Log(M)))
```
### Example Output
```
O(n)      24.038118866627283
O(m)      23.94367484916167
O(nm)     29.02337995301113
O(nlogm)  25.22909794002936
```
