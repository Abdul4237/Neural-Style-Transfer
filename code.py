import tensorflow as tf
import os as os
from PIL import Image
import matplotlib as plt
from matplotlib.pyplot import imshow
import numpy as np

gpu = tf.config.list_physical_devices('GPU')
for i in gpu:
    tf.config.experimental.set_memory_growth(i, True)
vgg=tf.keras.applications.VGG19(
    include_top=False,
    input_shape=(720,1280,3),
    weights=(r'C:\Users\wahee\OneDrive\Desktop\Neural Style Tranfer\vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
)
vgg.Trainable=False

def compute_content_cost(content_output,generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    _,n_H,n_W,n_C=a_G.get_shape().as_list()

    a_C_unrolled=tf.reshape(a_C,(_,n_H*n_W,n_C))
    a_G_unrolled=tf.reshape(a_G,(_,n_H*n_W,n_C))

    content_layer_loss=(1/(4*n_H*n_W*n_C))* tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))
    return content_layer_loss

def gram_matrix(A):
    gram=tf.matmul(A,tf.transpose(A))
    return gram


def compute_style_cost_layer(a_S,a_G):
    _,n_H,n_W,n_C=a_G.get_shape().as_list()
    
    a_S1=tf.transpose(tf.reshape(a_S,(n_H*n_W,n_C)))
    a_G1=tf.transpose(tf.reshape(a_G,(n_H*n_W,n_C)))
    
    G_S=gram_matrix(a_S1)
    G_G=gram_matrix(a_G1)

    style_layer_loss= (1/((2*n_C*n_H*n_W)))* tf.reduce_sum(tf.square(tf.subtract(G_S,G_G)))
    return style_layer_loss

STYLE_LAYERS=[
    ('block1_conv1',0.175),
    ('block2_conv1',0.175),
    ('block3_conv1',0.2),
    ('block4_conv1',0.2),
    ('block5_conv1',0.25)]

def compute_style_cost(style_image_output,generated_image_output,STYLE_LAYERS=STYLE_LAYERS):
    J_style=0
    a_S=style_image_output[:-1]
    a_G=generated_image_output[:-1]

    for i,weight in zip(range(len(a_S)),STYLE_LAYERS):
        J_style_layer=compute_style_cost_layer(a_S[i],a_G[i])
        J_style+= weight[1]* J_style_layer

    return J_style 
def total_cost(J_content, J_style,J_noise, alpha = 10, beta = 40,gamma=4):
    J_total=alpha*J_content + beta*J_style + J_noise*gamma
    return J_total


content_image=np.array(Image.open(r"C:\Users\wahee\OneDrive\Desktop\Neural Style Tranfer\images\EARTH.png"))
content_image=content_image[:,:,:-1]
content_image=tf.constant(np.reshape(content_image,((1,)+content_image.shape)))

style_image=np.array(Image.open(r"C:\Users\wahee\OneDrive\Desktop\Neural Style Tranfer\images\style.jpg"))
style_image=tf.constant(np.reshape(style_image,((1,)+style_image.shape)))


content_layer = [('block5_conv4', 1)]

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model    

vgg_model_activations=get_layer_outputs(vgg,layer_names=STYLE_LAYERS+content_layer)
preprocessed_content =  tf.image.resize(tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32)),(720,1280),preserve_aspect_ratio=True)
#preprocessed_content=tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_activations(preprocessed_content)
preprocessed_style =  tf.image.resize(tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32)),(720,1280),preserve_aspect_ratio=True)
#preprocessed_style=tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_activations(preprocessed_style)


generated_image=tf.Variable(tf.image.convert_image_dtype(preprocessed_content,tf.float32))
noise=tf.random.uniform(tf.shape(generated_image),-0.25,0.25)
generated_image=tf.add(generated_image,noise)
generated_image=tf.clip_by_value(generated_image, clip_value_min=0.0,clip_value_max=1.0)
a_G=vgg_model_activations(generated_image)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0) 
def tensor_to_image(tensor):
    tensor = tensor *255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

tensor_to_image(generated_image).show()
tensor_to_image(preprocessed_style).show()

optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
def train_step(generatedimage):
    with tf.GradientTape() as tape:
        tape.watch(generatedimage)
        a_G=vgg_model_activations(generatedimage)
        J_style=compute_style_cost(a_S,a_G)
        J_content=compute_content_cost(a_C,a_G)
        noise=tf.image.total_variation(generatedimage)
        J=total_cost(J_content,J_style,noise)
    grad=tape.gradient(J,generatedimage)
    optimizer.apply_gradients([(grad, generatedimage)])
    generatedimage.assign(clip_0_1(generatedimage))
    

epochs=20001
generated_image=tf.Variable(generated_image)
for i in range(epochs):
    train_step(generated_image)
    if i % 1000 == 0 and i!=0:
        print(f"Epoch {i} ")
        image = tensor_to_image(generated_image)
        image.show()
        image.save(f"Epoch{i},lr=0.010,nr=4.jpg")
        
