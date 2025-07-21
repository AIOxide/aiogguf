/*!
 * GGUF Parser Error Types
 */

use thiserror::Error;

pub type Result<T> = std::result::Result<T, GgufError>;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid GGUF magic number: expected 'GGUF', found {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid value type: {0}")]
    InvalidValueType(u32),

    #[error("Invalid quantization type: {0}")]
    InvalidQuantizationType(u32),

    #[error("Metadata key not found: {0}")]
    MetadataKeyNotFound(String),

    #[error("Invalid metadata value type for key '{key}': expected {expected}, found {found}")]
    InvalidMetadataValueType {
        key: String,
        expected: String,
        found: String,
    },

    #[error("String is not valid UTF-8: {0}")]
    InvalidUtf8(#[from] std::string::FromUtf8Error),

    #[error("Unexpected end of file")]
    UnexpectedEof,

    #[error("Invalid tensor dimensions")]
    InvalidTensorDimensions,

    #[error("Model configuration incomplete: missing {0}")]
    IncompleteModelConfig(String),
}