def select_from_web(link):
    from PIL import Image
    import requests
    im = Image.open(requests.get(link, stream=True).raw)
    array = tf.keras.preprocessing.image.img_to_array(im)
    input_image, ratio = prepare_image(array)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    label_encoder.visualize_detections(
        im,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
        
    )
    predict = dict(zip(class_names, detections.nmsed_scores[0][:num_detections])) 
    print(predict)