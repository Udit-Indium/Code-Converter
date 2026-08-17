import os
import time
import uuid
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

HOST = os.environ["DATABRICKS_HOST"]
TOKEN = os.environ["DATABRICKS_API_KEY"]
USER=os.environ["USER_ID"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type" : "application/json"
}

def execute_pyspark_script(python_script_path:str):
    file_name = f"generated_{uuid.uuid4().hex}.py"
    workspace_path = f"/Workspace/Users/{USER}@shell.com/Drafts/{file_name}"
    with open(python_script_path, "r") as file:
        code=file.read()
    timeout = 600
    print(code)

    try:
        upload_payload = {
            "path":workspace_path,
            "format":"SOURCE",
            "language":"PYTHON",
            "overwrite":True,
            "content":base64.b64encode(code.encode()).decode()
        }

        r = requests.post(
            f"{HOST}/api/2.0/workspace/import",
            headers=HEADERS,
            json=upload_payload
        )

        print(r.status_code)
        print(r.text)
        r.raise_for_status()

        if not r.ok:
            print(r.status_code, r.text)

        print("script_uploaded")

        submit_payload = {
            "run_name":"varification",
            "tasks":[
                {
                    "task_key":"execute",
                    "notebook_task":{
                        "notebook_path":workspace_path,
                    },
                    "environment_key":"default_python"
                }

            ],
            "environments":[
                {
                    "environment_key":"default_python",
                    "spec":{
                        "environment_version":"4"
                    }
                }
            ]
        }

        r = requests.post(
            f"{HOST}/api/2.2/jobs/runs/submit",
            headers=HEADERS,
            json=submit_payload
        )

        r.raise_for_status()

        run_id = r.json()["run_id"]
        print("Job submitted with run_id", run_id)

        start = time.time()

        while True:
            r = requests.get(
                f"{HOST}/api/2.2/jobs/runs/get",
                headers=HEADERS,
                params={"run_id":run_id},
            )

            r.raise_for_status()
            info = r.json()
            print(info)
            task = info["tasks"][0]
            task_run_id = task["run_id"]
            state = task["state"]["life_cycle_state"]
            print(state)

            if state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
                break

            if time.time()-start>timeout:
                return {
                    "status":"TIMEOUT",
                    "run_id":run_id
                }

            time.sleep(5)

        r = requests.get(
            f"{HOST}/api/2.2/jobs/runs/get-output",
            headers=HEADERS,
            params={"run_id":task_run_id}
        )
        output = r.json()
        return {
            "status":info["state"].get("result_state"),
            "life_cycle_state":state,
            "run_id":run_id,
            "error":output.get("error"),
            "metadata":output,
        }

    finally:
        try:
            requests.post(
                f"{HOST}/api/2.0/workspace/delete",
                headers=HEADERS,
                json={
                    "path":workspace_path,
                    "recursive":False,
                },
            )
            print("workspace_deleted")
        except Exception as ex:
            print("cleanup failed")

if __name__ == "__main__":
    # Take the script from the command line. The previous hardcoded path was
    # absolute, Windows-only and specific to one machine, so this entry point
    # could not run anywhere else.
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <path-to-pyspark-script.py>")
    print(execute_pyspark_script(sys.argv[1]))
