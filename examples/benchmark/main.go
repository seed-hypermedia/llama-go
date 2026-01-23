// Benchmark example generates random inputs and runs embedding batches to measure throughput.
//
// Usage:
//
//	benchmark -m embedding-model.gguf -b 100 -c 128
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		modelPath = flag.String("m", "embedding-model.gguf", "path to embedding model")
		batchSize = flag.Int("b", 100, "number of random strings to embed")
		gpuLayers = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		context   = flag.Int("c", 128, "context size")
		seed      = flag.Int64("seed", 0, "random seed (0 uses time-based seed)")
	)
	flag.Parse()

	if *batchSize <= 0 {
		fmt.Fprintln(os.Stderr, "batch size must be > 0")
		os.Exit(1)
	}
	if *context <= 0 {
		fmt.Fprintln(os.Stderr, "context size must be > 0")
		os.Exit(1)
	}

	// Load model with embeddings enabled
	fmt.Printf("Loading embedded model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithGPULayers(*gpuLayers),
		llama.WithMMap(true),
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

	randSeed := *seed
	if randSeed == 0 {
		randSeed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(randSeed))

	texts := make([]string, *batchSize)
	for i := 0; i < *batchSize; i++ {
		texts[i] = randomText(rng, *context)
	}

	fmt.Printf("Running embedding benchmark with %d random strings (approx length %d).\n", *batchSize, *context)
	benchmarkStart := time.Now()
	_, err = ctx.GetEmbeddingsBatch(texts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating batch embeddings: %v\n", err)
		os.Exit(1)
	}
	benchmarkElapsed := time.Since(benchmarkStart)

	fmt.Printf("Batch embedding elapsed: %s\n", benchmarkElapsed)
}

func randomText(rng *rand.Rand, targetLen int) string {
	var sb strings.Builder
	sb.Grow(targetLen + targetLen/4)

	for sb.Len() < targetLen {
		wordLen := 3 + rng.Intn(8)
		for i := 0; i < wordLen; i++ {
			sb.WriteByte(byte('a' + rng.Intn(26)))
		}
		sb.WriteByte(' ')
	}

	text := sb.String()
	if len(text) > targetLen {
		return text[:targetLen]
	}
	return text
}
