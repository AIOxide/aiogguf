/*!
 * GGUF Tensor Information and Quantization Types
 */

use crate::error::{GgufError, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek};

/// Quantization types supported by GGUF
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u32)]
pub enum QuantizationType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
}

impl QuantizationType {
    /// Check if this is a quantized type (not full precision)
    pub fn is_quantized(&self) -> bool {
        !matches!(self, QuantizationType::F32 | QuantizationType::F16 | QuantizationType::F64)
    }

    /// Get the bits per weight for this quantization type
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantizationType::F32 => 32.0,
            QuantizationType::F16 => 16.0,
            QuantizationType::F64 => 64.0,
            QuantizationType::Q4_0 | QuantizationType::Q4_1 => 4.5,
            QuantizationType::Q5_0 | QuantizationType::Q5_1 => 5.5,
            QuantizationType::Q8_0 | QuantizationType::Q8_1 => 8.5,
            QuantizationType::Q2_K => 2.5625,
            QuantizationType::Q3_K => 3.4375,
            QuantizationType::Q4_K => 4.5,
            QuantizationType::Q5_K => 5.5,
            QuantizationType::Q6_K => 6.5625,
            QuantizationType::Q8_K => 8.5,
            QuantizationType::IQ2_XXS => 2.0625,
            QuantizationType::IQ2_XS => 2.3125,
            QuantizationType::IQ3_XXS => 3.0625,
            QuantizationType::IQ1_S => 1.5625,
            QuantizationType::IQ4_NL => 4.5,
            QuantizationType::IQ3_S => 3.4375,
            QuantizationType::IQ2_S => 2.5,
            QuantizationType::IQ4_XS => 4.25,
            QuantizationType::I8 => 8.0,
            QuantizationType::I16 => 16.0,
            QuantizationType::I32 => 32.0,
            QuantizationType::I64 => 64.0,
            QuantizationType::IQ1_M => 1.75,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            QuantizationType::F32 => "32-bit float",
            QuantizationType::F16 => "16-bit float",
            QuantizationType::F64 => "64-bit float",
            QuantizationType::Q4_0 => "4-bit quantized (symmetric)",
            QuantizationType::Q4_1 => "4-bit quantized (asymmetric)",
            QuantizationType::Q5_0 => "5-bit quantized (symmetric)",
            QuantizationType::Q5_1 => "5-bit quantized (asymmetric)",
            QuantizationType::Q8_0 => "8-bit quantized (symmetric)",
            QuantizationType::Q8_1 => "8-bit quantized (asymmetric)",
            QuantizationType::Q2_K => "2-bit K-quantized",
            QuantizationType::Q3_K => "3-bit K-quantized",
            QuantizationType::Q4_K => "4-bit K-quantized",
            QuantizationType::Q5_K => "5-bit K-quantized",
            QuantizationType::Q6_K => "6-bit K-quantized",
            QuantizationType::Q8_K => "8-bit K-quantized",
            QuantizationType::IQ2_XXS => "2-bit IMatrix (extra small)",
            QuantizationType::IQ2_XS => "2-bit IMatrix (small)",
            QuantizationType::IQ3_XXS => "3-bit IMatrix (extra small)",
            QuantizationType::IQ1_S => "1-bit IMatrix (small)",
            QuantizationType::IQ4_NL => "4-bit IMatrix (non-linear)",
            QuantizationType::IQ3_S => "3-bit IMatrix (small)",
            QuantizationType::IQ2_S => "2-bit IMatrix (small)",
            QuantizationType::IQ4_XS => "4-bit IMatrix (extra small)",
            QuantizationType::I8 => "8-bit integer",
            QuantizationType::I16 => "16-bit integer",
            QuantizationType::I32 => "32-bit integer",
            QuantizationType::I64 => "64-bit integer",
            QuantizationType::IQ1_M => "1-bit IMatrix (medium)",
        }
    }
}

impl TryFrom<u32> for QuantizationType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(QuantizationType::F32),
            1 => Ok(QuantizationType::F16),
            2 => Ok(QuantizationType::Q4_0),
            3 => Ok(QuantizationType::Q4_1),
            6 => Ok(QuantizationType::Q5_0),
            7 => Ok(QuantizationType::Q5_1),
            8 => Ok(QuantizationType::Q8_0),
            9 => Ok(QuantizationType::Q8_1),
            10 => Ok(QuantizationType::Q2_K),
            11 => Ok(QuantizationType::Q3_K),
            12 => Ok(QuantizationType::Q4_K),
            13 => Ok(QuantizationType::Q5_K),
            14 => Ok(QuantizationType::Q6_K),
            15 => Ok(QuantizationType::Q8_K),
            16 => Ok(QuantizationType::IQ2_XXS),
            17 => Ok(QuantizationType::IQ2_XS),
            18 => Ok(QuantizationType::IQ3_XXS),
            19 => Ok(QuantizationType::IQ1_S),
            20 => Ok(QuantizationType::IQ4_NL),
            21 => Ok(QuantizationType::IQ3_S),
            22 => Ok(QuantizationType::IQ2_S),
            23 => Ok(QuantizationType::IQ4_XS),
            24 => Ok(QuantizationType::I8),
            25 => Ok(QuantizationType::I16),
            26 => Ok(QuantizationType::I32),
            27 => Ok(QuantizationType::I64),
            28 => Ok(QuantizationType::F64),
            29 => Ok(QuantizationType::IQ1_M),
            _ => Err(GgufError::InvalidQuantizationType(value)),
        }
    }
}

/// Information about a tensor in a GGUF file
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub quantization_type: QuantizationType,
    pub offset: u64,
}

impl TensorInfo {
    /// Read all tensor information from a reader
    pub fn read_all<R: Read + Seek>(reader: &mut R, tensor_count: u64) -> Result<Vec<Self>> {
        let mut tensors = Vec::with_capacity(tensor_count as usize);

        for _ in 0..tensor_count {
            // Read tensor name
            let name = {
                let mut name_len_buf = [0u8; 8];
                reader.read_exact(&mut name_len_buf)?;
                let name_len = u64::from_le_bytes(name_len_buf);

                let mut name_buf = vec![0u8; name_len as usize];
                reader.read_exact(&mut name_buf)?;
                String::from_utf8(name_buf)?
            };

            // Read number of dimensions
            let n_dimensions = {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                u32::from_le_bytes(buf)
            };

            if n_dimensions > 4 {
                return Err(GgufError::InvalidTensorDimensions);
            }

            // Read dimensions
            let mut dimensions = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                let mut dim_buf = [0u8; 8];
                reader.read_exact(&mut dim_buf)?;
                dimensions.push(u64::from_le_bytes(dim_buf));
            }

            // Read quantization type
            let quantization_type = {
                let mut type_buf = [0u8; 4];
                reader.read_exact(&mut type_buf)?;
                QuantizationType::try_from(u32::from_le_bytes(type_buf))?
            };

            // Read tensor data offset
            let offset = {
                let mut offset_buf = [0u8; 8];
                reader.read_exact(&mut offset_buf)?;
                u64::from_le_bytes(offset_buf)
            };

            tensors.push(TensorInfo {
                name,
                dimensions,
                quantization_type,
                offset,
            });
        }

        Ok(tensors)
    }

    /// Calculate the size of this tensor in bytes
    pub fn size_bytes(&self) -> u64 {
        if self.dimensions.is_empty() {
            return 0;
        }

        let element_count: u64 = self.dimensions.iter().product();
        let bits_per_element = self.quantization_type.bits_per_weight();
        
        // Round up to nearest byte
        ((element_count as f64 * bits_per_element as f64) / 8.0).ceil() as u64
    }

    /// Get tensor shape as a formatted string
    pub fn shape_string(&self) -> String {
        format!("[{}]", self.dimensions.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "))
    }

    /// Check if this is a weight tensor (not bias or other auxiliary tensors)
    pub fn is_weight_tensor(&self) -> bool {
        self.name.contains("weight") || 
        self.name.contains("embed") ||
        self.name.contains("norm") ||
        self.name.contains("attn") ||
        self.name.contains("ffn") ||
        self.name.contains("mlp")
    }

    /// Get the layer number if this tensor belongs to a specific layer
    pub fn layer_number(&self) -> Option<u32> {
        // Common patterns: "layers.0.weight", "blocks.15.norm", etc.
        if let Some(layers_pos) = self.name.find("layers.") {
            let start = layers_pos + 7; // "layers.".len()
            if let Some(dot_pos) = self.name[start..].find('.') {
                if let Ok(layer_num) = self.name[start..start + dot_pos].parse::<u32>() {
                    return Some(layer_num);
                }
            }
        }
        
        if let Some(blocks_pos) = self.name.find("blocks.") {
            let start = blocks_pos + 7; // "blocks.".len()
            if let Some(dot_pos) = self.name[start..].find('.') {
                if let Ok(layer_num) = self.name[start..start + dot_pos].parse::<u32>() {
                    return Some(layer_num);
                }
            }
        }
        
        None
    }
}