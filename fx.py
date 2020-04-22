def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return str(int(n * multiplier) / multiplier)

def pred_classify(datagen, pred_model):
    # Get first batch
    images, labels = datagen.next()

    img = images[0]

    preds = model.predict(np.expand_dims(img, axis = 0))

    ps = preds[0]
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)
    classes = ['benign', 'malignant']
    print(ps)

    print('Probabilities:')
    print(pd.DataFrame({'Class Label': ['benign', 'malignant'], 'Probabilties': ps}))

    if ps[0] > 0.6:
        print("This area of skin appears to be benign, but you should still check it out at the doctor's if you aren't sure!!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")
    else:
        print("That's no ordinary abnormality, you should definitely check it out at the doctor's!")
        print(truncate(ps[0]*100, 1) + "% benign, " + truncate(ps[1]*100, 1) + "% malignant.")

def print_preds(datagen, pred_model):
    # Get first batch
    images, labels = datagen.next()

    img = images[0]
    preds = model.predict(np.expand_dims(img, axis = 0))
    ps = preds[0]

    print(img)
    # Swap the class name and its numeric representation
    classes = sorted(datagen.class_indices, key=datagen.class_indices.get)

    print('Probabilities:')
    print(pd.DataFrame({'Class Label': classes, 'Probabilties': ps}))
    if labels[0][0] == 1.0:
        print('Actual Label: Benign')
    elif labels[0][1] == 1.0:
        print('Actual Label: Malignant')
    else:
        print('Actual Label: Label Error')
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.squeeze())
    ax1.axis('off')
    ax2.set_yticks(np.arange(len(classes)))
    ax2.barh(np.arange(len(classes)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticklabels(classes, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
