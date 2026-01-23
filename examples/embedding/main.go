// Embedding example demonstrates generating text embeddings for semantic tasks.
//
// This program loads a GGUF embedding model and computes vector representations
// of input text. Embeddings are useful for semantic search, clustering, similarity
// comparison, and other machine learning tasks that require numerical representations
// of text.
//
// Usage:
//
//	embedding -m embedding-model.gguf -t "text to embed"
//
// The model must be loaded with embedding support enabled (WithEmbeddings option).
// Not all models support embeddings - check model documentation before use. Typical
// embedding models include sentence transformers and specialised embedding variants.
//
// The example demonstrates:
//   - Loading models in embedding mode
//   - Generating embeddings from text
//   - Inspecting embedding vector properties
//   - Computing basic embedding statistics
//
// Embeddings can be used for:
//   - Semantic search (finding similar documents)
//   - Clustering (grouping related texts)
//   - Classification (ML training features)
//   - Similarity scoring (comparing text meaning)
//
// Output includes the embedding dimension, sample values, and basic magnitude
// statistics to verify the embedding generation succeeded.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"time"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		modelPath = flag.String("m", "embedding-model.gguf", "path to embedding model")
		text      = flag.String("t", "Hello world", "text to get embeddings for")
		gpuLayers = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		context   = flag.Int("c", 128, "context size")
	)
	flag.Parse()
	os.Setenv("LLAMA_LOG", "error") // Quiet mode
	// Load model with embeddings enabled
	fmt.Printf("Loading embedded model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
		llama.WithSilentLoading(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	// Create context with embedding support
	ctx, err := model.NewContext(
		llama.WithContext(*context),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithEmbeddings(),
		llama.WithF16Memory(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating context: %v\n", err)
		os.Exit(1)
	}
	defer ctx.Close()

	fmt.Printf("Model loaded successfully.\n")
	fmt.Printf("Getting embeddings for: %s\n", *text)

	// Generate embeddings
	embeddingStart := time.Now()
	embeddings, err := ctx.GetEmbeddings(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating embeddings: %v\n", err)
		os.Exit(1)
	}
	embeddingElapsed := time.Since(embeddingStart)

	fmt.Printf("\nEmbeddings generated successfully!\n")
	fmt.Printf("Vector dimension: %d\n", len(embeddings))

	postStart := time.Now()
	magnitude := float32(0.0)
	for _, val := range embeddings {
		magnitude += val * val
	}

	norm := float32(math.Sqrt(float64(magnitude)))
	if norm > 0 {
		for i := range embeddings {
			embeddings[i] /= norm
		}
	}

	meanSquared := magnitude / float32(len(embeddings)) // Mean squared (pre-normalization)
	fmt.Printf("Mean squared magnitude: %.6f\n", meanSquared)
	fmt.Printf("L2 norm (pre-normalization): %.6f\n", norm)
	magnitude = 0.0
	fmt.Printf("Embeddings:[")
	for _, val := range embeddings {
		fmt.Printf("%.8f, ", val)
		magnitude += val * val
	}
	fmt.Printf("]\n")
	norm = float32(math.Sqrt(float64(magnitude)))
	fmt.Printf("L2 norm (post-normalization): %.6f\n", norm)
	postElapsed := time.Since(postStart)

	fmt.Printf("\nTiming:\n")
	fmt.Printf("  Embedding generation: %s\n", embeddingElapsed)
	fmt.Printf("  Post-processing: %s\n", postElapsed)
}
