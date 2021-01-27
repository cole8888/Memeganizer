# Memeganizer
*This is a work in progress, I am still increasing the size of the dataset and implementing new features. I also need to figure out what the dependencies are and how to install this since I never originally planned to publish this.*

Neural network powered meme-organizing tool to help separate memes from images that are not memes.

It can help you free up space on your mobile device and make backing up your photos into the proper folders easier.  

Please note that no neural network is perfect. As of writing, the model is >99.6% accurate according to my validation tests and first hand experience, but it can perform better or worse than this depending on the workload.

There are various things in images or types of images which have a higher chance of confusing the model, these include:
- Captions, text bars, filters or stickers.
- "Deep fried" memes or "Nuked" memes.
- Collages which have text. (Especially large text, may confuse it with a starterpack meme)
- Graphs or charts (Many memes have these)
- Images with extreme aspect ratios. (Very tall or very wide, loses detail at downscaled resolution)
- Video game screenshots. (Especially with large text)
- Imgur's "image not found" placeholder picture. (Classified as a meme)
- Images of the real world which prominently include a meme in the foreground (printed, drawn, painted or on a screen)
- The bare images from a common meme template without the text or meme layout.

Screenshots of text posts from Reddit, Twitter, Tumblr, Instagram, Facebook, 4chan, etc... will generally considered memes since this model cannot actually undstand the content of the text itself, it looks at the overall structure of the image. Think of it like looking at posts in a language you don't understand, you know there is text there but you have no idea what it is saying or whether it is intended to be funny.

These cases will not always fail, I've made the dataset quite robust and it should be able to handle some deviations.

Keep in mind this model is not able to understand that things like this are memes:
![Image](https://github.com/cole8888/Memeganizer/raw/main/Unable_to_Understand.png)

Since the amount of prior context and inference required to understand why that picture is a meme is not yet possible to communicate to a neural network.

It can understand things like this though:

Meme
![Image](https://github.com/cole8888/Memeganizer/raw/main/Can_Detect_Meme.jpg "Meme")

Not-Meme
![Image](https://github.com/cole8888/Memeganizer/raw/main/Can_Detect_Not-Meme.jpg "Not-Meme")

The dataset so far includes 200000 Memes and 200000 Not-Memes totalling 204.8GB. It was mostly scraped from Reddit and Instagram. I have tried as hard as I can to have a diverse dataset ranging from Biology memes to Star Wars memes to Soccer memes and everything in between. The Not-Memes group has a very diverse pool of images ranging from selfies to plants to furniture. This group needs to be as diverse as possible.

I manually sorted the first 4000 or so and have been using intermediate models to help do most of the initial sorting and build a larger and larger dataset. I also occasionally let the model sort a copy of it's own training dataset to let me see what templates / types it's struggling with, this also helps me find errors in the dataset. I manually go through all the new images I add after the intermediate model has sorted them to ensure that no mistakes are being made.
