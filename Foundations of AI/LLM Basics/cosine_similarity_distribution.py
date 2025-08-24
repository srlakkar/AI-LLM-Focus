import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
np.random.seed(42)

def generate_cosine_similarity_distribution(n_samples=10000, vector_dim=100):
    """
    Generate distribution of cosine similarity between random vectors.
    
    Parameters:
    n_samples: Number of random vector pairs to generate
    vector_dim: Dimension of the random vectors
    """
    
    similarities = []
    
    for _ in range(n_samples):
        # Generate two random vectors
        v1 = np.random.randn(vector_dim)
        v2 = np.random.randn(vector_dim)
        
        # Calculate cosine similarity
        # Note: cosine_similarity returns a matrix, we take the [0,1] element
        similarity = cosine_similarity([v1], [v2])[0, 0]
        similarities.append(similarity)
    
    return np.array(similarities)

# Generate the distribution
similarities = generate_cosine_similarity_distribution(n_samples=10000, vector_dim=100)

# Create the visualization
plt.figure(figsize=(12, 8))

# Plot 1: Histogram with density curve
plt.subplot(2, 2, 1)
plt.hist(similarities, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
plt.axvline(np.median(similarities), color='green', linestyle='--', label=f'Median: {np.median(similarities):.3f}')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.title('Distribution of Cosine Similarity\n(Random Vectors)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Box plot
plt.subplot(2, 2, 2)
plt.boxplot(similarities, vert=True)
plt.ylabel('Cosine Similarity')
plt.title('Box Plot of Cosine Similarity')
plt.grid(True, alpha=0.3)

# Plot 3: Q-Q plot to check normality
plt.subplot(2, 2, 3)
from scipy import stats
stats.probplot(similarities, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normal Distribution)')
plt.grid(True, alpha=0.3)

# Plot 4: Cumulative distribution
plt.subplot(2, 2, 4)
sorted_similarities = np.sort(similarities)
cumulative_prob = np.arange(1, len(sorted_similarities) + 1) / len(sorted_similarities)
plt.plot(sorted_similarities, cumulative_prob, linewidth=2)
plt.xlabel('Cosine Similarity')
plt.ylabel('Cumulative Probability')
plt.title('Cumulative Distribution Function')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("Summary Statistics:")
print(f"Number of samples: {len(similarities)}")
print(f"Mean: {np.mean(similarities):.4f}")
print(f"Median: {np.median(similarities):.4f}")
print(f"Standard deviation: {np.std(similarities):.4f}")
print(f"Min: {np.min(similarities):.4f}")
print(f"Max: {np.max(similarities):.4f}")
print(f"Range: {np.max(similarities) - np.min(similarities):.4f}")

# Theoretical expectation for random vectors in high dimensions
print(f"\nTheoretical expectation for random vectors: ~0")
print(f"Empirical mean: {np.mean(similarities):.4f}")

# Check if distribution is approximately normal
from scipy.stats import shapiro
statistic, p_value = shapiro(similarities)
print(f"\nShapiro-Wilk test for normality:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Distribution is {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)") 