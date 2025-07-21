/*!
 * GGUF Parser Tests
 */

use crate::*;
use std::path::Path;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tinyllama_gguf_parsing() {
        let model_path = "/Users/dex/source/aioxide/models/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q8_0.gguf";
        
        if !Path::new(model_path).exists() {
            println!("Skipping test - model file not found: {}", model_path);
            return;
        }

        let gguf_file = GgufFile::from_file(model_path).expect("Failed to parse GGUF file");
        
        // Verify header
        assert!(gguf_file.header.is_valid());
        println!("Header: {:?}", gguf_file.header);
        
        // Print all metadata keys for debugging
        println!("Metadata keys:");
        for key in gguf_file.metadata.data.keys() {
            println!("  {}", key);
        }
        
        // Verify we can extract model config
        let config = gguf_file.model_config().expect("Failed to extract model config");
        println!("Model config: {:?}", config);
        
        // Basic assertions
        assert_eq!(config.architecture, "llama");
        assert!(config.vocab_size > 0);
        assert!(config.context_length > 0);
        assert!(config.block_count > 0);
        
        // Check quantization
        assert!(gguf_file.is_quantized());
        let quant_types = gguf_file.quantization_types();
        println!("Quantization types: {:?}", quant_types);
        assert!(quant_types.contains(&QuantizationType::Q8_0));
        
        // Print some tensor info
        println!("Total tensors: {}", gguf_file.tensors.len());
        println!("Total size: {} MB", gguf_file.total_size() / 1024 / 1024);
        
        for (i, tensor) in gguf_file.tensors.iter().take(5).enumerate() {
            println!("Tensor {}: {} {} {:?} ({} bytes)", 
                i, tensor.name, tensor.shape_string(), 
                tensor.quantization_type, tensor.size_bytes());
        }
    }

    #[test]
    fn test_liquidai_gguf_parsing() {
        let model_path = "/Users/dex/.lmstudio/models/LiquidAI/LFM2-1.2B-GGUF/LFM2-1.2B-Q4_0.gguf";
        
        if !Path::new(model_path).exists() {
            println!("Skipping test - model file not found: {}", model_path);
            return;
        }

        let gguf_file = GgufFile::from_file(model_path).expect("Failed to parse GGUF file");
        
        // This model has no config.json, so all config must come from GGUF
        let config = gguf_file.model_config().expect("Failed to extract model config");
        println!("LiquidAI config: {:?}", config);
        
        // Check quantization
        let quant_types = gguf_file.quantization_types();
        println!("LiquidAI quantization types: {:?}", quant_types);
        assert!(quant_types.contains(&QuantizationType::Q4_0));
    }

    #[test]
    fn test_mistral_multifile_gguf() {
        let model_path = "/Users/dex/.lmstudio/models/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-GGUF/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf";
        
        if !Path::new(model_path).exists() {
            println!("Skipping test - model file not found: {}", model_path);
            return;
        }

        let gguf_file = GgufFile::from_file(model_path).expect("Failed to parse GGUF file");
        
        let config = gguf_file.model_config().expect("Failed to extract model config");
        println!("Mistral config: {:?}", config);
        
        // Check quantization
        let quant_types = gguf_file.quantization_types();
        println!("Mistral quantization types: {:?}", quant_types);
        assert!(quant_types.contains(&QuantizationType::Q4_K));
        
        // Check for vision projector file
        let vision_projector_path = "/Users/dex/.lmstudio/models/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-GGUF/mmproj-Mistral-Small-3.2-24B-Instruct-2506-F16.gguf";
        if Path::new(vision_projector_path).exists() {
            let vision_gguf = GgufFile::from_file(vision_projector_path).expect("Failed to parse vision projector GGUF");
            println!("Vision projector tensors: {}", vision_gguf.tensors.len());
        }
    }

    #[test]
    fn test_quantization_type_bits() {
        assert_eq!(QuantizationType::F32.bits_per_weight(), 32.0);
        assert_eq!(QuantizationType::Q4_0.bits_per_weight(), 4.5);
        assert_eq!(QuantizationType::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(QuantizationType::Q2_K.bits_per_weight(), 2.5625);
        
        assert!(QuantizationType::Q4_0.is_quantized());
        assert!(!QuantizationType::F32.is_quantized());
    }

    #[test]
    fn test_tensor_layer_parsing() {
        let tensor = TensorInfo {
            name: "layers.15.attn.weight".to_string(),
            dimensions: vec![4096, 4096],
            quantization_type: QuantizationType::Q4_0,
            offset: 0,
        };
        
        assert_eq!(tensor.layer_number(), Some(15));
        assert!(tensor.is_weight_tensor());
        assert_eq!(tensor.shape_string(), "[4096, 4096]");
    }
}