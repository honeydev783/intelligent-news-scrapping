from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.session import get_db

router = APIRouter()



@router.get("/article/{id}")
def article(id: int, db: Session = Depends(get_db)):
    result = db.execute(text("SELECT * FROM articles WHERE id = :id"), {"id" : id})
    return result.mappings().first()