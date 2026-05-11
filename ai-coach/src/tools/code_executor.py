"""Code execution sandbox."""

# src/tools/code_executor.py

import docker
from typing import Tuple, Optional


class CodeExecutionError(Exception):
    pass


class CodeExecutor:
    """Executes Python code in a Docker sandbox."""
    
    def __init__(self):
        self.client = docker.from_env()
        
    def execute(self, 
                code: str,
                timeout: int = 10,
                user_id: str = "guest") -> Tuple[str, str]:
        """
        Execute Python code in a sandboxed container.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            user_id: User ID for isolation (used in container name)
        
        Returns:
            Tuple of (stdout, stderr)
        
        Raises:
            CodeExecutionError: If execution fails or times out
        """
        container_name = f"code-executor-{user_id}"
        
        try:
            # Stop and remove existing container if it exists
            try:
                container = self.client.containers.get(container_name)
                container.stop()
                container.remove()
            except docker.errors.NotFound:
                pass
            
            # Create and run a new container with the code
            container = self.client.containers.run(
                "python:3.10-slim",
                command="python -c \"{code}\"",
                name=container_name,
                mem_limit="128m",
                cpu_shares=512,
                network_disabled=True,
                detach=True
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            
            # Get output
            stdout = container.logs(stdout=True).decode("utf-8")
            stderr = container.logs(stderr=True).decode("utf-8")
            
            # Stop and remove container
            container.stop()
            container.remove()
            
            if result["StatusCode"] != 0:
                raise CodeExecutionError(f"Execution failed:\n{stderr}")
            
            return stdout.strip(), stderr.strip()
            
        except docker.errors.APIError as e:
            raise CodeExecutionError(f"Docker API error: {e}")
        except Exception as e:
            raise CodeExecutionError(f"Execution error: {e}")
        finally:
            # Clean up any running containers
            try:
                container = self.client.containers.get(container_name)
                container.stop()
                container.remove()
            except:
                pass