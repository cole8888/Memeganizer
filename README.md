# Memeganizer
Neural network powered meme-organizing tool to help separate memes from images that are not memes.

It can help you free up space on your mobile device and make backing up your photos into the proper folders easier.  

Please note that no neural network is perfect. This one tends to be 99.2% accurate according to my validation tests and first hand experience, but it can perform better or worse than this depending on the workload.

There are various edgecases which may confuse the model, these include:
- Snapchat caption text bars and filters.
- "Deep fried" memes or "Nuked" memes
- Collages which have text. (Especially with large text)
- Digital paintings with low detail levels.
- Distorted or rotated memes.
- Images with extreme aspect ratios. (Very tall or very wide)
- Video game screenshots. (Especially with large text)
- Uncommon or new templates featuring things which are not meme related with no textbar or text.
- Imgur's "image not found" placeholder picture.

Screenshots of text posts from Reddit, Twitter, Tumblr, Instagram, Facebook, 4chan, etc... will generally considered memes since this model cannot actually undstand the content of the text itself, it looks at the overall structure of the image.

The dataset so far includes 51500 Memes and 51500 Not-Memes totalling 47GB and it was mostly scraped from Reddit and Instagram. I have tried as hard as I can to have a diverse dataset ranging from biologymemes to PrequelMemes and everything in between. I manually sorted the first 4000 or so and have been using intermediate models to help do most of the initial sorting and build a larger and larger dataset. I also occasionally let the model sort a copy of it's own training dataset to let me see what templates / types it's struggling with, this also helps me find errors in the dataset.
