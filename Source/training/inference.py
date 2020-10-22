# inference.py
import numpy as np
import json
 
def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    if context.request_content_type in ('application/x-npy', "application/npy"):
        import numpy as np
        # If we're an array of numpy objects, handle that
        #if isinstance(data[0], np.ndarray):
        #    data = [x.tolist() for x in data]
        #else:
        #    data = data.tolist()
        #print(data)
        data = np.asarray(data).astype(float).tolist()
        return json.dumps({
            "instances": data
        })
        #data = {'instances': numpy.asarray(np_array).astype(float).tolist()}


    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type