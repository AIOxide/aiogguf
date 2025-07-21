# aiogguf

GGUF (Generic Graph Universal Format) parser for AI model metadata extraction.

## Features

- **GGUF v3 specification compliant** - Parses headers, metadata, and tensor information
- **Model configuration extraction** - Automatically extracts vocab size, layers, attention heads, etc.
- **Quantization detection** - Supports all quantization types (Q4_0, Q8_0, K-quants, IMatrix)
- **Multi-file support** - Handles split GGUF files and vision projectors
- **Architecture support** - LLaMA, Mistral, Qwen, Phi3, Gemma, Mixtral, CodeLlama
- **Zero dependencies** - Only uses `serde`, `serde_json`, and `thiserror`

## Usage

```rust
use aiogguf::GgufFile;

// Parse a GGUF file
let gguf_file = GgufFile::from_file("model.gguf")?;

// Extract model configuration
let config = gguf_file.model_config()?;
println!("Architecture: {}", config.architecture);
println!("Parameters: {}", config.estimated_param_count());

// Check quantization
if gguf_file.is_quantized() {
    println!("Quantization types: {:?}", gguf_file.quantization_types());
}

// Inspect tensors
for tensor in &gguf_file.tensors {
    println!("{}: {} {:?}", tensor.name, tensor.shape_string(), tensor.quantization_type);
}
```

## Tested Models

- **TinyLlama-1.1B-Chat-v1.0** (Q8_0 quantization)
- **LiquidAI/LFM2-1.2B-GGUF** (Q4_0, no config.json)
- **Mistral-Small-3.2-24B** (Q4_K_M, multi-file with vision projector)

## License

MIT OR Apache-2.0