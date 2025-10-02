import numpy as np

def scalarize(weights, 
              objectives_components, 
              ideal_point=None, 
              nadir_point = None, 
              scalarization_method="linear", 
              normalize = True,
              smoothness = 0.01
              ):
    """
    Scalarizes the objectives components using the specified scalarization method.
    
    Args:
        weights (np.ndarray): Weights for each objective.
        objectives_components (np.ndarray): Components of the objectives to be scalarized.
        scalarization_method (str): Method to use for scalarization ('linear' or other).
        
    Returns:
        float: Scalarized value.
    """

    epsilon = 1e-5
    ideal_point = ideal_point + epsilon * np.ones_like(ideal_point) if ideal_point is not None else None

    # === Normalization ===
    if normalize and ideal_point is not None:
        objectives_components = objectives_components / (ideal_point + epsilon)
        ideal_point = np.ones_like(ideal_point)

    if scalarization_method == "linear":
        return np.sum(weights * objectives_components)
    
    elif scalarization_method == "tchebycheff":
        assert ideal_point is not None, "Ideal point must be provided for Tchebycheff scalarization."
        shortfalls = np.maximum(ideal_point - objectives_components, 0)  # fix: no abs for maximization
        weighted_shortfalls = weights * shortfalls
        scalarized_value = np.max(weighted_shortfalls)
        return -scalarized_value  # fix: smaller shortfall => larger scalar value
    
    elif scalarization_method == "augmented_tchebycheff":
        assert ideal_point is not None, "Ideal point must be provided for augmented Tchebycheff scalarization."
        rho = 1e-6
        shortfalls = np.maximum(ideal_point - objectives_components, 0)
        weighted_shortfalls = weights * shortfalls
        scalarized_value = np.max(weighted_shortfalls) + rho * np.sum(weighted_shortfalls)
        return -scalarized_value  # fix
    
    elif scalarization_method == "pbi":
        assert ideal_point is not None, "Ideal point must be provided for PBI scalarization."
        theta = 5.0
        diff = ideal_point - objectives_components 
        norm_w = np.linalg.norm(weights)
        if norm_w == 0:
            raise ValueError("Weight vector has zero norm.")
        w = weights / norm_w
        d1 = np.dot(diff, w)
        projection = d1 * w
        d2 = np.linalg.norm(diff - projection)
        return -(d1 + theta * d2)
    
    elif scalarization_method == "smooth_tchebycheff":
        assert ideal_point is not None
        u = smoothness  # smoothness parameter
        diffs = weights * np.maximum(ideal_point - objectives_components, 0)
        scaled = diffs / u
        max_scaled = np.max(scaled)
        stch = u * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
        return -stch

    # Add other scalarization methods as needed
    raise ValueError("Invalid scalarization method")  # Default case if no valid method is provided