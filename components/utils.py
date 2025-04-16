def convert_for_matlab(data, context=None):
    """Recursively convert Python dict to MATLAB-friendly format."""
    if isinstance(data, dict):
        converted = {}
        for key, value in data.items():
            # Ensure the key is a string.
            original_key = key
            if not isinstance(key, str):
                key = str(key)

            current_key = key
            current_context = context

            # Update context for nested structures (before key conversion!)
            if key == "metrics_segment":
                current_context = "metrics_segment_thresholds"
            elif context == "metrics_segment_thresholds":
                current_context = "metrics_segment_classes"
            elif key == "metrics_sample":
                current_context = "metrics_sample_metrics"
            elif key == "label_distribution":
                current_context = "label_distribution_values"

            # Handle key conversions based on the current context
            if isinstance(key, str):
                # If we're processing threshold keys (from "metrics_segment"),
                # convert them from (e.g., "0.1") to a valid field name (e.g., "t10").
                if context == "metrics_segment_thresholds":
                    try:
                        num = float(key)
                        num_int = int(round(num * 100))
                        current_key = f"t{num_int:02d}"  # Ensure two-digit format
                    except ValueError:
                        pass
                # For contexts where keys represent class labels or metrics,
                # prepend a letter to ensure the key is a valid MATLAB field name.
                elif current_context in {
                    "metrics_segment_classes",
                    "metrics_sample_metrics",
                    "label_distribution_values",
                }:
                    # If the key consists of digits only, convert it.
                    if key.isdigit():
                        current_key = f"c{key}"
            # If key conversion did not result in a valid string, fall back to the original string key.
            current_key = str(current_key)

            # Recurse into sub-structures
            converted_value = convert_for_matlab(value, context=current_context)
            converted[current_key] = converted_value

        return converted

    elif isinstance(data, list):
        return [convert_for_matlab(item, context=context) for item in data]
    else:
        return data
