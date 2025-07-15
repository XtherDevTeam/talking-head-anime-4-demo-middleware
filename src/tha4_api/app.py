import asyncio
import threading
from tha4_api import logger
import io
import livekit
import flask_cors
import flask
import tha4_api.live2DRealtimeSessionManager
import typing
import json

app = flask.Flask(__name__)
flask_cors.CORS(app)
live2d_session_manager = tha4_api.live2DRealtimeSessionManager.Live2DRealtimeSessionManager()


def Result(status: bool = True, data: typing.Any = None):
    return {
        'status': status,
        'data': data
    }


@app.route('/establish_session', methods=['POST'])
def establish_session():
    configuration = flask.request.form.get('configuration')
    baseFps = flask.request.form.get('baseFps')
    avatar = flask.request.files.get('avatar')
    live2Dtoken = flask.request.form.get('live2Dtoken')
    livekitUrl = flask.request.form.get('livekitUrl')
    if not configuration or not avatar:
        return Result(False, 'Invalid request')
    
    try:
        configuration = json.loads(configuration)
    except json.JSONDecodeError:
        return Result(False, 'Invalid configuration')
    
    try:
        baseFps = int(baseFps)
    except (ValueError, TypeError):
        return Result(False, 'Invalid baseFps')
    
    avatar_blob = io.BytesIO()
    avatar.save(avatar_blob)
    
    session_name = live2d_session_manager.createSession(configuration, avatar_blob.getvalue(), baseFps) # if cache is found, baseFps will be overwritten by the cached value
    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(live2d_session_manager.getSession(session_name).start(live2Dtoken, livekitUrl, loop))
        
    threading.Thread(target=runner).start()
    
    return Result(True, session_name)


@app.route('/switch_state', methods=['POST'])
def switch_state():
    session_name = flask.request.json.get('sessionName')
    state = flask.request.json.get('state')
    if not session_name or not state:
        return Result(False, 'Invalid request')
    
    session = live2d_session_manager.getSession(session_name)
    if not session:
        return Result(False, 'Invalid session')
    
    session.renderer.switch_state(state)
    
    return Result(True)


@app.route('/shutdown_session', methods=['POST'])
def shutdown_session():
    session_name = flask.request.json.get('sessionName')
    if not session_name:
        return Result(False, 'Invalid request')
    
    live2d_session_manager.shutdownSession(session_name)
    
    return Result(True)


@app.route('/compile', methods=['POST'])
def compile():
    configuration = flask.request.form.get('configuration')
    baseFps = flask.request.form.get('baseFps')
    avatar = flask.request.files.get('avatar')
    if not configuration or not avatar and not baseFps:
        return Result(False, 'Invalid request')
    
    try:
        configuration = json.loads(configuration)
    except json.JSONDecodeError:
        return Result(False, 'Invalid configuration')
    
    try:
        baseFps = int(baseFps)
        avatar_blob = io.BytesIO()
        avatar.save(avatar_blob)
        tha4_api.live2DRealtimeSessionManager.compileForConfiguration(configuration, avatar_blob.getvalue(), baseFps)
    except Exception as e:
        return Result(False, str(e))
    
    return Result(True)


if __name__ == '__main__':
    logger.Logger.log(f'Starting THA4 API middleware server on port 3623')
    app.run(debug=False, host='0.0.0.0', port=3623)