def backpropagation_network(inputs: np.ndarray, perceptrons: int, layers: int) -> float:
    """
    Construye una red totalmente conectada con 'layers' capas y 'perceptrons' neuronas por capa,
    con pesos iniciales aleatorios. Realiza la propagación hacia adelante y retorna la salida
    agregada de la capa de salida (promedio de las activaciones si hay más de una neurona).
    Nota: Esta implementación inicializa aleatoriamente pesos y sesgos y realiza solo la
    propagación hacia adelante (retorna 'a' de la(s) neurona(s) de salida).
    """
    # Mensaje para identificar que entramos exitosamente a la función
    print("corriendo red de retropropagación con los siguientes parámetros:\n- entradas: {inputs}\n- perceptrones por capa: {perceptrons}\n- capas: {layers}\n".format(inputs=inputs, perceptrons=perceptrons, layers=layers))

    # Aseguramos tipos y reproducibilidad
    inputs = np.asarray(inputs, dtype=float)
    random.seed(42)
    np.random.seed(42)

    # layers_data será lista de capas; cada capa es lista de Perceptron
    layers_data: list[list[Perceptron]] = []

    for layer_idx in range(layers):
        layer_perceptrons: list[Perceptron] = []

        # Entradas a la capa actual: entradas originales si es la primera capa,
        # o las activaciones (a) de la capa anterior.
        if layer_idx == 0:
            prev_values = list(inputs)
        else:
            prev_values = [p.a for p in layers_data[-1]]

        for _ in range(perceptrons):
            # Crear InputData con cada valor de entrada y un peso aleatorio
            p_inputs = [InputData(x=val, w=random.uniform(-1.0, 1.0)) for val in prev_values]
            # Sesgo aleatorio
            b = random.uniform(-1.0, 1.0)
            # Crear perceptrón (la clase Perceptron calculará z y a en su constructor)
            p = Perceptron(p_inputs, b)
            layer_perceptrons.append(p)

        layers_data.append(layer_perceptrons)

    # Capa de salida: si tiene una sola neurona devolvemos su 'a', si tiene varias devolvemos el promedio
    output_layer = layers_data[-1]
    output_activations = [p.a for p in output_layer]

    # Calculo final de salida (a)
    if len(output_activations) == 1:
        network_output = float(output_activations[0])
    else:
        network_output = float(np.mean(output_activations))

    # Return de la función: cálculo de "a" de la capa de salida
    return network_output
