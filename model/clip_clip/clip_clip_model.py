import tensorflow as tf
from tf_keras.models import Model, load_model
from transformers import TFCLIPModel
from model.constrastive_loss import ContrastiveLoss

class ClipCLIP(Model):  # ClipCLIP references is the exact same as StreetCLIP references, but with a different model and processor
    def __init__(self, load_path=None, **kwargs):
        super(ClipCLIP, self).__init__(**kwargs)

        if load_path is not None:
            self.clip_model = load_model(load_path, custom_objects={"ContrastiveLoss": ContrastiveLoss})
        else:
            # Descending order of size
            self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
            # self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            # self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch16")  # DOESN'T EXIST AS TFCLIPModel
            # self.clip_model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    @classmethod
    def from_save(cls, load_path, **kwargs):
        return cls(load_path=load_path, **kwargs)

    def infer(self, inputs, ret_key="logits_per_image", ret_np=False):
        outputs = self.clip_model(inputs)  # **inputs (changed for compatibility)

        if ret_key is None:
            ret = outputs
        else:
            ret = outputs[ret_key]  # outputs.get(ret_key, outputs)
        # logits_per_image = outputs["logits_per_image"]  # .logits_per_image (changed for compatibility)

        if ret_np:
            return ret.numpy()

        return ret

    def call(self, inputs, ret_key="logits_per_image", ret_np=False):
        return self.infer(inputs, ret_key=ret_key, ret_np=ret_np)

    def clip_fit(self, *args, **kwargs):
        self.clip_model.fit(*args, **kwargs)

    def clip_compile(self, *args, **kwargs):
        self.clip_model.compile(*args, **kwargs)

    def cosine_sim_with_scale(self, x, y):
        mul = tf.matmul(x, y, transpose_b=True)
        scale = tf.exp(self.clip_model.clip.logit_scale)

        return mul * scale

    # A custom fit method that accumulates embeddings before computing loss to effectively increase batch size without increasing memory usage (at the cost of speed)
    def custom_fit(self, train_data, initial_epoch, final_epoch, callbacks, effective_batch_size):
        callbacks = tf.keras.callbacks.CallbackList(callbacks, add_history=True, add_progbar=True, model=self, epochs=final_epoch, initial_epoch=initial_epoch, verbose=1)
        callbacks._progbar.stateful_metrics = ["loss"]  # work-around instead of just adding progbar to callbacklist

        train_data = train_data.prefetch(tf.data.AUTOTUNE)

        callbacks.on_train_begin()
        for epoch in range(initial_epoch, final_epoch):
            callbacks.on_epoch_begin(epoch)

            # Training
            accumulated_embeds = [[], []]
            with tf.GradientTape() as tape:
                for step, (x_batch, y_batch) in enumerate(train_data):
                    callbacks.on_train_batch_begin(step)

                    predictions = self(x_batch, training=True, ret_key=None)
                    accumulated_embeds[0].append(predictions["image_embeds"])
                    accumulated_embeds[1].append(predictions["text_embeds"])

                    callbacks.on_train_batch_end(step, logs={"loss": -1})

                    used_batch_size = len(x_batch["pixel_values"])
                    if len(accumulated_embeds[0]) >= effective_batch_size // used_batch_size:
                        image_embeds = tf.concat(accumulated_embeds[0], axis=0)
                        text_embeds = tf.concat(accumulated_embeds[1], axis=0)

                        sim = self.cosine_sim_with_scale(image_embeds, text_embeds)
                        loss = self.compute_loss(y=tf.zeros_like(sim), y_pred=sim)

                        break

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables), jit_compile=True)

            logs = {"loss": loss.numpy()}

            callbacks.on_epoch_end(epoch, logs=logs)

        callbacks.on_train_end()
