/*!
 * GGUF File Format Parser
 * 
 * Pure Rust implementation for parsing GGUF (Generic Graph Universal Format) files.
 * Focused on extracting model metadata and configuration for AI model inference.
 */

mod error;
mod header;
mod metadata;
mod tensor;
mod types;

#[cfg(test)]
mod tests;

pub use error::{GgufError, Result};
pub use header::GgufHeader;
pub use metadata::{GgufMetadata, ModelConfig};
pub use tensor::{TensorInfo, QuantizationType};
pub use types::{GgufValue, GgufValueType};

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// Main GGUF file parser
#[derive(Debug)]
pub struct GgufFile {
    pub header: GgufHeader,
    pub metadata: GgufMetadata,
    pub tensors: Vec<TensorInfo>,
}

impl GgufFile {
    /// Parse a GGUF file from a file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::from_reader(&mut reader)
    }

    /// Parse a GGUF file from a reader
    pub fn from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Parse header
        let header = GgufHeader::read(reader)?;
        
        // Parse metadata
        let metadata = GgufMetadata::read(reader, header.metadata_kv_count)?;
        
        // Parse tensor information
        let tensors = TensorInfo::read_all(reader, header.tensor_count)?;
        
        Ok(Self {
            header,
            metadata,
            tensors,
        })
    }

    /// Extract model configuration for inference
    pub fn model_config(&self) -> Result<ModelConfig> {
        ModelConfig::from_metadata(&self.metadata)
    }

    /// Get total file size in bytes
    pub fn total_size(&self) -> u64 {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }

    /// Check if this is a quantized model
    pub fn is_quantized(&self) -> bool {
        self.tensors.iter().any(|t| t.quantization_type.is_quantized())
    }

    /// Get all quantization types used in this model
    pub fn quantization_types(&self) -> Vec<QuantizationType> {
        let mut types: Vec<_> = self.tensors
            .iter()
            .map(|t| t.quantization_type)
            .collect();
        types.sort();
        types.dedup();
        types
    }
}