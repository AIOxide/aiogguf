/*!
 * GGUF Header Parsing
 */

use crate::error::{GgufError, Result};
use std::io::{Read, Seek};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const SUPPORTED_VERSION: u32 = 3;

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Read GGUF header from a reader
    pub fn read<R: Read + Seek>(reader: &mut R) -> Result<Self> {
        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        // Read version
        let mut version_buf = [0u8; 4];
        reader.read_exact(&mut version_buf)?;
        let version = u32::from_le_bytes(version_buf);

        if version != SUPPORTED_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        // Read tensor count
        let mut tensor_count_buf = [0u8; 8];
        reader.read_exact(&mut tensor_count_buf)?;
        let tensor_count = u64::from_le_bytes(tensor_count_buf);

        // Read metadata key-value count
        let mut metadata_kv_count_buf = [0u8; 8];
        reader.read_exact(&mut metadata_kv_count_buf)?;
        let metadata_kv_count = u64::from_le_bytes(metadata_kv_count_buf);

        Ok(Self {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    /// Get header size in bytes
    pub fn size(&self) -> usize {
        4 + 4 + 8 + 8 // magic + version + tensor_count + metadata_kv_count
    }

    /// Check if this is a valid GGUF file
    pub fn is_valid(&self) -> bool {
        self.magic == GGUF_MAGIC && self.version == SUPPORTED_VERSION
    }
}