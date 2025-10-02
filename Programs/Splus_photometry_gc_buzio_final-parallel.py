error_map[valid_weight] = 1.0 / np.sqrt(weight_data[valid_weight])
error_flux = np.sqrt(np.sum(error_in_aperture**2))
