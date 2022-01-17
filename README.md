# video-autosize

**The challenge:** I saw an [interesting challenge](https://twitter.com/matthen2/status/1483160741222006788?s=20) posed in a Tweet. In this challenge, you are streaming a video pixel-by-pixel, but do not know the dimensions (width and height) of the video. How do you figure out the width and height automatically?

# My approach

To succeed at the challenge, we must settle on a distribution `p(X)` over videos `X`. For example, if the videos are randomly sampled TV static, we cannot infer anything about the original resolution since all pixels are independent of each other. However, if we assume the videos are from some natural distribution (e.g. videos of the natural world taken on a camera), then larger-scale statistics will likely tip off the resolution of the video.

If this sounds like a machine learning problem, that's because it can be setup as one. In particular, we can only test algorithms in the context of some video distribution. We can represent the video distribution as a dataset of videos, and test an algorithm's success rate using a validation set of videos. We don't necessarily have to train our algorithm on a training set of videos--we could hardcode algorithms as well. But we still can't know if we've coded something reasonable until we test it on a validation set.
