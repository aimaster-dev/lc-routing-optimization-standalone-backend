from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pydantic import BaseModel
from typing import List

router = APIRouter()

class Coordinates(BaseModel):
    Latitude: float
    Longitude: float

class Stop(BaseModel):
    Latitude: float
    Longitude: float
    CURRENT_CONTAINER_SIZE: int
    SERVICE_WINDOW_TIME: float
    SERVICE_TYPE_CD: str
    PERM_NOTES: str

class RouteData(BaseModel):
    Haul: Coordinates
    LandFill: Coordinates
    Stops: List[Stop]

@router.post("/route-map-old")
async def get_route(data: RouteData):
    """
    Returns the corresponding HTML map based on the route_type and location_id.
    """
    from services.route_optimizer_old import generate_route_map

    await generate_route_map(data.model_dump())
    print("asdfasdfasdfasdf")
    manual_file_name = f"maps/manual_map.html"
    # optimal_file_name = f"maps/optimal_map_old.html"
    optimal_file_name = f"maps/optimal_map.html"
    try:
        with open(manual_file_name, "r") as manual_file:
            manual_html_content = manual_file.read()
        with open(optimal_file_name, "r") as optimal_file:
            optimal_html_content = optimal_file.read()
        return {
            "html_manual": manual_html_content,
            "html_optimal": optimal_html_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/route-map/comparison/location")
async def get_route():
    """
    Returns the corresponding HTML map based on the route_type and location_id.
    """
    manual_file_name = f"maps/manual_map.html"
    optimal_file_name = f"maps/optimal_map.html"
    try:
        with open(manual_file_name, "r") as manual_file:
            manual_html_content = manual_file.read()
        with open(optimal_file_name, "r") as optimal_file:
            optimal_html_content = optimal_file.read()
        return {
            "html_manual": manual_html_content,
            "html_optimal": optimal_html_content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))