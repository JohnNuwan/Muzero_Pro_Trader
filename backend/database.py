from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite Database
DATABASE_URL = "sqlite:///./gemini.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket = Column(Integer, unique=True, index=True)
    symbol = Column(String, index=True)
    type = Column(String) # BUY / SELL
    lot = Column(Float)
    price_open = Column(Float)
    price_close = Column(Float, nullable=True)
    sl = Column(Float)
    tp = Column(Float)
    profit = Column(Float, default=0.0)
    status = Column(String) # OPEN, CLOSED
    open_time = Column(DateTime, default=datetime.utcnow)
    close_time = Column(DateTime, nullable=True)
    magic = Column(Integer)
    comment = Column(String, nullable=True)
    strategy = Column(String, nullable=True) # TREND, REVERSION, SNIPER, etc.

class Event(Base):
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, index=True) # INFO, WARNING, ERROR, EVOLUTION, NEWS
    message = Column(Text)
    symbol = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Snapshot(Base):
    __tablename__ = "snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    balance = Column(Float)
    equity = Column(Float)
    daily_profit = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

# --- INIT DB ---
def check_schema_updates():
    """Check and apply schema updates (migrations)"""
    try:
        # Create a temporary connection to check schema
        with engine.connect() as connection:
            # Check if 'strategy' column exists in 'trades'
            # This is a raw SQL check for SQLite
            result = connection.execute(text("PRAGMA table_info(trades)")).fetchall()
            columns = [row[1] for row in result]
            
            if "strategy" not in columns:
                print("Migrating DB: Adding 'strategy' column to 'trades'...")
                connection.execute(text("ALTER TABLE trades ADD COLUMN strategy VARCHAR"))
                connection.commit()
                print("Migration Complete.")
    except Exception as e:
        print(f"Schema check failed: {e}")

def init_db():
    check_schema_updates()
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
