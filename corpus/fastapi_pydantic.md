# FastAPI and Pydantic Reference

## FastAPI Overview

FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. Key features:
- Automatic OpenAPI / Swagger documentation
- Request and response validation via Pydantic
- Async support with `async def`
- Dependency injection system

## Installation

```bash
pip install fastapi uvicorn
```

## Creating an Application

```python
from fastapi import FastAPI

app = FastAPI(title="My API", version="1.0.0")

@app.get("/")
def root():
    return {"message": "Hello World"}
```

Run with:
```bash
uvicorn main:app --reload --port 8000
```

## Request Body with Pydantic

```python
from pydantic import BaseModel, Field
from fastapi import FastAPI

class Item(BaseModel):
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    description: str | None = None

@app.post("/items/")
def create_item(item: Item):
    return item
```

## Path and Query Parameters

```python
@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

## HTTP Status Codes and Error Handling

```python
from fastapi import HTTPException, status

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    return db[item_id]
```

## File Uploads

```python
from fastapi import UploadFile, File
from typing import List

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        results.append({"filename": file.filename, "size": len(content)})
    return results
```

## CORS Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Response Models

```python
class ItemResponse(BaseModel):
    id: int
    name: str
    created: bool

@app.post("/items/", response_model=ItemResponse, status_code=201)
def create_item(item: Item):
    ...
```

## Background Tasks

```python
from fastapi import BackgroundTasks

def send_email(email: str):
    # runs after response is sent
    ...

@app.post("/notify/")
def notify(background_tasks: BackgroundTasks, email: str):
    background_tasks.add_task(send_email, email)
    return {"message": "Notification queued"}
```

---

## Pydantic v2 Models

Pydantic provides data validation using Python type annotations.

### Basic Model

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class User(BaseModel):
    name: str
    age: int = Field(..., ge=0, le=150)
    email: Optional[str] = None
    tags: List[str] = []
```

### Field Constraints

```python
class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price in USD")
    quantity: int = Field(default=0, ge=0)
```

### Model Configuration

```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    name: str
```

### Validators

```python
from pydantic import field_validator

class User(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v.lower()
```
