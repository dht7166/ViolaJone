# Viola Jones algorithm for face detection
## Implemented by
- Hoan Tran - hdtran@buffalo.edu

- Hoang Lan - hoanglan@buffalo.edu


## Algorithm overview
The violajones algorithm is implemented based on the cascading
of adaboosts of weak haar feature classifier.

At the lowest level, we use basic haar like features as classifier.
These low-level haar feature are then augmented using adaboost for better accuracy.
The algorithm futher uses cascading for faster detection time, as well as lower false positive.
Each cascade layer is a adaboosted classifier, designed to maximize detection rate while lowering false positive rate (by changing the threshold).
Then, at each layer, the current window will be tested if it is a face or not. If it is not, it is quickly discarded.

## Implementation details
### Feature extraction
I set the base resolution for detection to be 17x17. Based on my implementation, there will be a total of 60000+ features.
Using sklearn.feature_selection, I can choose the top 5000 features.

### Training the adaboost classifier.
First, I computed the integral image for all images in the dataset (after face extracted).
Then, using the feature chosen in the previous part, I calculated a matrix of size images x features (or Ix5000).
Then following the outline in section 3 of the paper, I implemented the adaboost classifier, using this matrix. I implemented the weak classifier learning using the method described in section 3.1 of the paper.

## Results on FDDB


## Further improvement
If more time is allowed, we can do a full feature training instead of a selected 5000 features.
