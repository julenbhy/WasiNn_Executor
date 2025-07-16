use serde::{Deserialize, Deserializer, Serialize};
use std::{collections::HashMap, fmt::Debug};

#[derive(PartialEq, Clone)]
pub enum ActivationResponseStatus {
    Success = 0,
    ApplicationError = 1,
    // ActionDeveloperError = 2,
    // WhiskInternalError = 3,
}

impl Serialize for ActivationResponseStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use ActivationResponseStatus::*;

        match self {
            Success => serializer.serialize_str("success"),
            ApplicationError => serializer.serialize_str("application error"),
            // ActionDeveloperError => serializer.serialize_str("action developer error"),
            // WhiskInternalError => serializer.serialize_str("whisk internal error"),
        }
    }
}

#[derive(Serialize)]
pub struct ActivationResponse {
    status: ActivationResponseStatus,
    status_code: u8,
    success: bool,
    result: serde_json::Value,
}

impl ActivationResponse {
    pub fn new(result: Result<serde_json::Value, serde_json::Value>) -> Self {
        match result {
            Ok(ok) => {
                let status = ActivationResponseStatus::Success;
                Self {
                    success: true,
                    status_code: status.clone() as u8,
                    status,
                    result: ok,
                }
            }
            Err(err) => {
                let status = ActivationResponseStatus::ApplicationError;
                Self {
                    success: false,
                    status_code: status.clone() as u8,
                    status,
                    result: err,
                }
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ActivationInit {
    pub value: ActivationInitInner,
}

#[derive(Serialize, Deserialize)]
pub struct ActivationInitInner {
    pub name: String,
    pub main: String,
    pub code: String,
    pub binary: bool,
    pub env: HashMap<String, String>,
    pub annotations: ActionCapabilities,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct ActionCapabilities {
    pub dir: Option<String>,
    pub net_access: Option<bool>,
    pub model_urls: Option<Vec<String>>,
}

impl Debug for ActivationInitInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationInitInner")
            .field("name", &self.name)
            .field("main", &self.main)
            .field("binary", &self.binary)
            .field("env", &self.env)
            .field("annotations", &self.annotations)
            .finish()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationContext {
    pub value: serde_json::Value,
    pub namespace: String,
    pub action_name: String,
    pub api_host: Option<String>,
    pub api_key: Option<String>,
    pub activation_id: String,
    pub transaction_id: String,
    #[serde(deserialize_with = "str_to_u64")]
    pub deadline: u64,
}

fn str_to_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let buf = String::deserialize(deserializer)?;

    buf.parse::<u64>().map_err(serde::de::Error::custom)
}

pub struct WasmAction<M> {
    pub module: M,
    pub capabilities: ActionCapabilities,
}

pub trait WasmRuntime: Clone {
    fn initialize(
        &self,
        container_id: String,
        capabilities: ActionCapabilities,
        module: Vec<u8>,
    ) -> anyhow::Result<()>;

    fn run(
        &self,
        container_id: &str,
        parameters: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value>;

    fn destroy(&self, container_id: &str);
}