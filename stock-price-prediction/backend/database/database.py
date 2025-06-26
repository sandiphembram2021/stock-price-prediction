from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base
import os
from typing import Generator
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for SQLite operations"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Create database directory if it doesn't exist
            db_dir = "backend/database"
            os.makedirs(db_dir, exist_ok=True)
            database_url = f"sqlite:///{db_dir}/stock_prediction.db"
        
        self.database_url = database_url
        
        # Create engine with SQLite-specific settings
        self.engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get synchronous database session"""
        return self.SessionLocal()

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """Dependency function for FastAPI to get database session"""
    return db_manager.get_session()
