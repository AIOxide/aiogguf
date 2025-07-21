/*!
 * GGUF Metadata Parsing and Model Configuration Extraction
 */

use crate::error::{GgufError, Result};
use crate::types::{GgufValue, GgufValueType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// GGUF metadata container
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub data: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    /// Read metadata from a reader
    pub fn read<R: Read + Seek>(reader: &mut R, kv_count: u64) -> Result<Self> {
        let mut data = HashMap::new();

        for _ in 0..kv_count {
            // Read key
            let key = {
                let mut key_len_buf = [0u8; 8];
                reader.read_exact(&mut key_len_buf)?;
                let key_len = u64::from_le_bytes(key_len_buf);

                let mut key_buf = vec![0u8; key_len as usize];
                reader.read_exact(&mut key_buf)?;
                String::from_utf8(key_buf)?
            };

            // Read value type
            let value_type = {
                let mut type_buf = [0u8; 4];
                reader.read_exact(&mut type_buf)?;
                GgufValueType::try_from(u32::from_le_bytes(type_buf))?
            };

            // Read value
            let value = GgufValue::read(reader, value_type)?;
            data.insert(key, value);
        }

        Ok(Self { data })
    }

    /// Get a metadata value by key
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.data.get(key)
    }

    /// Get a required metadata value by key
    pub fn get_required(&self, key: &str) -> Result<&GgufValue> {
        self.data
            .get(key)
            .ok_or_else(|| GgufError::MetadataKeyNotFound(key.to_string()))
    }

    /// Get a string value
    pub fn get_string(&self, key: &str) -> Result<&str> {
        self.get_required(key)?.as_string()
    }

    /// Get an optional string value
    pub fn get_string_opt(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_string().ok())
    }

    /// Get a u32 value
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        self.get_required(key)?.as_u32()
    }

    /// Get an optional u32 value
    pub fn get_u32_opt(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(|v| v.as_u32().ok())
    }

    /// Get a u64 value
    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.get_required(key)?.as_u64()
    }

    /// Get an optional u64 value
    pub fn get_u64_opt(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64().ok())
    }

    /// Get a f32 value
    pub fn get_f32(&self, key: &str) -> Result<f32> {
        self.get_required(key)?.as_f32()
    }

    /// Get an optional f32 value
    pub fn get_f32_opt(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(|v| v.as_f32().ok())
    }
}

/// Model configuration extracted from GGUF metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Basic architecture info
    pub architecture: String,
    pub vocab_size: u64,
    pub context_length: u64,
    
    // Layer configuration
    pub block_count: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    
    // Attention configuration
    pub attention_head_count: u32,
    pub attention_head_count_kv: Option<u32>,
    pub attention_layer_norm_rms_epsilon: Option<f32>,
    
    // Rope configuration
    pub rope_dimension_count: Option<u32>,
    pub rope_freq_base: Option<f32>,
    pub rope_scaling_type: Option<String>,
    
    // Tokenizer info
    pub tokenizer_ggml_model: Option<String>,
    pub tokenizer_ggml_tokens: Option<Vec<String>>,
    pub tokenizer_ggml_scores: Option<Vec<f32>>,
    pub tokenizer_ggml_token_type: Option<Vec<u32>>,
    
    // Chat template
    pub tokenizer_chat_template: Option<String>,
    
    // Additional metadata
    pub general_name: Option<String>,
    pub general_description: Option<String>,
    pub general_license: Option<String>,
}

impl ModelConfig {
    /// Extract model configuration from GGUF metadata
    pub fn from_metadata(metadata: &GgufMetadata) -> Result<Self> {
        // Architecture is required
        let architecture = metadata.get_string("general.architecture")?.to_string();
        
        // Use architecture-specific prefixes for parameter names
        let arch_prefix = format!("{architecture}.");
        
        // Required parameters - vocab_size can be inferred from tokenizer tokens
        let vocab_size = metadata.get_u64("general.vocab_size")
            .or_else(|_| metadata.get_u64(&format!("{arch_prefix}vocab_size")))
            .or_else(|_| {
                // Infer vocab_size from tokenizer tokens array length
                if let Some(GgufValue::Array(tokens)) = metadata.get("tokenizer.ggml.tokens") {
                    Ok(tokens.len() as u64)
                } else {
                    Err(GgufError::IncompleteModelConfig("vocab_size".to_string()))
                }
            })?;

        let context_length = metadata.get_u64("general.context_length")
            .or_else(|_| metadata.get_u64(&format!("{arch_prefix}context_length")))
            .map_err(|_| GgufError::IncompleteModelConfig("context_length".to_string()))?;

        let block_count = metadata.get_u32(&format!("{arch_prefix}block_count"))
            .map_err(|_| GgufError::IncompleteModelConfig("block_count".to_string()))?;

        let embedding_length = metadata.get_u32(&format!("{arch_prefix}embedding_length"))
            .map_err(|_| GgufError::IncompleteModelConfig("embedding_length".to_string()))?;

        let feed_forward_length = metadata.get_u32(&format!("{arch_prefix}feed_forward_length"))
            .map_err(|_| GgufError::IncompleteModelConfig("feed_forward_length".to_string()))?;

        let attention_head_count = metadata.get_u32(&format!("{arch_prefix}attention.head_count"))
            .map_err(|_| GgufError::IncompleteModelConfig("attention.head_count".to_string()))?;

        // Optional parameters
        let attention_head_count_kv = metadata.get_u32_opt(&format!("{arch_prefix}attention.head_count_kv"));
        let attention_layer_norm_rms_epsilon = metadata.get_f32_opt(&format!("{arch_prefix}attention.layer_norm_rms_epsilon"));
        
        let rope_dimension_count = metadata.get_u32_opt(&format!("{arch_prefix}rope.dimension_count"));
        let rope_freq_base = metadata.get_f32_opt(&format!("{arch_prefix}rope.freq_base"));
        let rope_scaling_type = metadata.get_string_opt(&format!("{arch_prefix}rope.scaling.type")).map(|s| s.to_string());
        
        // Tokenizer information
        let tokenizer_ggml_model = metadata.get_string_opt("tokenizer.ggml.model").map(|s| s.to_string());
        
        // TODO: Parse tokenizer arrays (tokens, scores, token_type)
        let tokenizer_ggml_tokens = None;
        let tokenizer_ggml_scores = None;
        let tokenizer_ggml_token_type = None;
        
        let tokenizer_chat_template = metadata.get_string_opt("tokenizer.chat_template").map(|s| s.to_string());
        
        // General metadata
        let general_name = metadata.get_string_opt("general.name").map(|s| s.to_string());
        let general_description = metadata.get_string_opt("general.description").map(|s| s.to_string());
        let general_license = metadata.get_string_opt("general.license").map(|s| s.to_string());

        Ok(ModelConfig {
            architecture,
            vocab_size,
            context_length,
            block_count,
            embedding_length,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon,
            rope_dimension_count,
            rope_freq_base,
            rope_scaling_type,
            tokenizer_ggml_model,
            tokenizer_ggml_tokens,
            tokenizer_ggml_scores,
            tokenizer_ggml_token_type,
            tokenizer_chat_template,
            general_name,
            general_description,
            general_license,
        })
    }

    /// Get model parameter count estimate
    pub fn estimated_param_count(&self) -> u64 {
        // Rough estimate based on transformer architecture
        let vocab_embedding = self.vocab_size * self.embedding_length as u64;
        let transformer_blocks = self.block_count as u64 * (
            // Self-attention weights
            4 * self.embedding_length as u64 * self.embedding_length as u64 +
            // Feed-forward weights  
            2 * self.embedding_length as u64 * self.feed_forward_length as u64 +
            // Layer norms
            2 * self.embedding_length as u64
        );
        let output_projection = self.vocab_size * self.embedding_length as u64;
        
        vocab_embedding + transformer_blocks + output_projection
    }

    /// Check if this is a supported architecture
    pub fn is_supported_architecture(&self) -> bool {
        matches!(self.architecture.as_str(), 
            "llama" | "mistral" | "qwen" | "qwen2" | "phi3" | "gemma" | "mixtral" | "codellama"
        )
    }
}