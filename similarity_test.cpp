#include <iostream>
#include <vector>
#include <cmath>

// Function to compute cosine similarity between two vectors
double compute_similarity(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Vectors must be of the same size." << std::endl;
        return -1.0;
    }
    
    double dot_product = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 == 0.0 || norm2 == 0.0) {
        return 0.0;
    }
    
    return dot_product / (norm1 * norm2);
}

int main() {
    // Example vectors for testing
    std::vector<double> vec1 = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> vec2 = {0.5, 0.4, 0.3, 0.2, 0.1};
    
    double similarity = compute_similarity(vec1, vec2);
    std::cout << "Cosine Similarity: " << similarity << std::endl;
    
    return 0;
}

