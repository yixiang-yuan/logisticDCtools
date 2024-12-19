# Large Scale Logistic Regression in High Dimension
STSCI 6520: Statistical Computing I capstone project.

The package provides functions for large scale, high-dimensional logistic regression analysis both with the entire data set or with data split across multiple cores for faster computation. The function uses some regularization in the list of LASSO, MCP or SCAD to tackle with high-dimensionality of the data, and will produce a sparse prediction with an active set specifying the predicted non-zero entries. The method for determining the combined active set is majority voting, and there are three different methods for aggregating the final estimates.

Please see the vignette for more information and a tutorial on how to use the package.
