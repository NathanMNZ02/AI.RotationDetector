from roboflow import Roboflow
rf = Roboflow(api_key="sj8dDAfXaFX1MVcFlX7w")
project = rf.workspace("nathanmnz").project("ai.fasterrcnn.rotationdetector")
version = project.version(4)
dataset = version.download("coco")
                
