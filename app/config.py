from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql://rag:rag@localhost:5434/belge_rag"
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    chat_model: str = "qwen2.5:7b"
    sim_threshold: float = 0.35


settings = Settings()
