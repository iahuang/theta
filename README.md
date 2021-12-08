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
    for i1 in [5, 10, 20, 40, 80, 160, 320, 640]
    for i2 in [5, 10, 20, 40, 80, 160, 320, 640]
)

# Benchmark our data
data = theta.compile_runtime_data(
    f=test_function,
    function_inputs=input_generator,
    min_iters=200,
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
O(n)      12.289116852356175
O(m)      12.285565621119934
O(nm)     21.331979668626328
O(nlogm)  12.91796806188527
```
Notice here that `O(nm)` has by far the highest correlation value.