/*!
 * GGUF Value Types
 */

use crate::error::{GgufError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// GGUF value type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(GgufValueType::Uint8),
            1 => Ok(GgufValueType::Int8),
            2 => Ok(GgufValueType::Uint16),
            3 => Ok(GgufValueType::Int16),
            4 => Ok(GgufValueType::Uint32),
            5 => Ok(GgufValueType::Int32),
            6 => Ok(GgufValueType::Float32),
            7 => Ok(GgufValueType::Bool),
            8 => Ok(GgufValueType::String),
            9 => Ok(GgufValueType::Array),
            10 => Ok(GgufValueType::Uint64),
            11 => Ok(GgufValueType::Int64),
            12 => Ok(GgufValueType::Float64),
            _ => Err(GgufError::InvalidValueType(value)),
        }
    }
}

/// GGUF value container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    /// Read a GGUF value from a reader
    pub fn read<R: Read + Seek>(reader: &mut R, value_type: GgufValueType) -> Result<Self> {
        match value_type {
            GgufValueType::Uint8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Uint8(buf[0]))
            }
            GgufValueType::Int8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int8(buf[0] as i8))
            }
            GgufValueType::Uint16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Uint16(u16::from_le_bytes(buf)))
            }
            GgufValueType::Int16 => {
                let mut buf = [0u8; 2];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int16(i16::from_le_bytes(buf)))
            }
            GgufValueType::Uint32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Uint32(u32::from_le_bytes(buf)))
            }
            GgufValueType::Int32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int32(i32::from_le_bytes(buf)))
            }
            GgufValueType::Float32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Float32(f32::from_le_bytes(buf)))
            }
            GgufValueType::Bool => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Bool(buf[0] != 0))
            }
            GgufValueType::String => {
                let length = {
                    let mut buf = [0u8; 8];
                    reader.read_exact(&mut buf)?;
                    u64::from_le_bytes(buf)
                };
                
                let mut string_buf = vec![0u8; length as usize];
                reader.read_exact(&mut string_buf)?;
                let string = String::from_utf8(string_buf)?;
                Ok(GgufValue::String(string))
            }
            GgufValueType::Array => {
                let array_type = {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    GgufValueType::try_from(u32::from_le_bytes(buf))?
                };
                
                let length = {
                    let mut buf = [0u8; 8];
                    reader.read_exact(&mut buf)?;
                    u64::from_le_bytes(buf)
                };
                
                let mut array = Vec::with_capacity(length as usize);
                for _ in 0..length {
                    array.push(GgufValue::read(reader, array_type)?);
                }
                Ok(GgufValue::Array(array))
            }
            GgufValueType::Uint64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Uint64(u64::from_le_bytes(buf)))
            }
            GgufValueType::Int64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Int64(i64::from_le_bytes(buf)))
            }
            GgufValueType::Float64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::Float64(f64::from_le_bytes(buf)))
            }
        }
    }

    /// Convert to specific type with validation
    pub fn as_u32(&self) -> Result<u32> {
        match self {
            GgufValue::Uint32(v) => Ok(*v),
            GgufValue::Uint64(v) => Ok(*v as u32),
            _ => Err(GgufError::InvalidMetadataValueType {
                key: "unknown".to_string(),
                expected: "u32".to_string(),
                found: format!("{:?}", self),
            }),
        }
    }

    pub fn as_u64(&self) -> Result<u64> {
        match self {
            GgufValue::Uint64(v) => Ok(*v),
            GgufValue::Uint32(v) => Ok(*v as u64),
            _ => Err(GgufError::InvalidMetadataValueType {
                key: "unknown".to_string(),
                expected: "u64".to_string(),
                found: format!("{:?}", self),
            }),
        }
    }

    pub fn as_string(&self) -> Result<&str> {
        match self {
            GgufValue::String(v) => Ok(v),
            _ => Err(GgufError::InvalidMetadataValueType {
                key: "unknown".to_string(),
                expected: "string".to_string(),
                found: format!("{:?}", self),
            }),
        }
    }

    pub fn as_f32(&self) -> Result<f32> {
        match self {
            GgufValue::Float32(v) => Ok(*v),
            _ => Err(GgufError::InvalidMetadataValueType {
                key: "unknown".to_string(),
                expected: "f32".to_string(),
                found: format!("{:?}", self),
            }),
        }
    }
}