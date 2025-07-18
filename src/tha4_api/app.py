import asyncio
import threading
from tha4_api import logger
import io
import livekit
import flask_cors
import flask
import tha4_api.live2DRealtimeSessionManager
import tha4_api.animation
import typing
import json
import flask_socketio
import PIL.Image

app = flask.Flask(__name__)
flask_cors.CORS(app)
socket = flask_socketio.SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, max_http_buffer_size=1e10)
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

    session_name = live2d_session_manager.createSession(configuration, avatar_blob.getvalue(
    ), baseFps)  # if cache is found, baseFps will be overwritten by the cached value

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(live2d_session_manager.getSession(
            session_name).start(live2Dtoken, livekitUrl, loop))

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
        return Result(False, f'Invalid session: {flask.session["auth"]}')

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
        tha4_api.live2DRealtimeSessionManager.compileForConfiguration(
            configuration, avatar_blob.getvalue(), baseFps)
    except Exception as e:
        return Result(False, str(e))

    return Result(True)


@socket.on('initialize', namespace='/live2d_development')
def on_initialize(data):
    print(f'Initializing session')
    flask.session['auth'] = flask.request.sid
    
    configuration, baseFps = data['configuration'], data['baseFps']

    # generate a 1x1 empty png image as the base image
    base_image = PIL.Image.new('RGBA', (512, 512))
    base_image_bytes = io.BytesIO()
    base_image.save(base_image_bytes, format='PNG')

    # create a new session
    session_name = live2d_session_manager.createSession(
        configuration, base_image_bytes.getvalue(), baseFps)
    
    live2d_session_manager.bindSessionSid(session_name, flask.session['auth'])
    
    socket.emit('avatar_required', flask.session['auth'], namespace='/live2d_development', room=flask.request.sid)
    

@socket.on('update_avatar', namespace='/live2d_development')
def on_update_avatar(data):
    avatar = data
    session = live2d_session_manager.getSessionBySid(flask.session['auth'])
    if not session:
        socket.emit('error', {
            'message': f'Invalid session: {flask.session["auth"]}'}, namespace='/live2d_development', room=flask.request.sid)
        return

    session.updateAvatar(avatar)
    
    
@socket.on('session_restored', namespace='/live2d_development')
def on_session_restored(data):
    print(f'restoring session: {data}')
    flask.session['auth'] = data


@socket.on('update_configuration', namespace='/live2d_development')
def on_update_configuration(data):
    configuration = data['configuration']

    session = live2d_session_manager.getSessionBySid(flask.session['auth'])
    if not session:
        socket.emit('error', {
            'message': f'Invalid session: {flask.session["auth"]}'}, namespace='/live2d_development', room=flask.request.sid)
        return

    session.renderer.configuration = tha4_api.animation.AnimationConfiguration(configuration)


@socket.on('render_animation_group', namespace='/live2d_development')
def on_render_animation_group(data):
    animation_group_id, state_name = data['animationGroupId'], data['stateName']

    session = live2d_session_manager.getSessionBySid(flask.session['auth'])
    if not session:
        socket.emit('error', {
            'message': f'Invalid session: {flask.session["auth"]}'}, namespace='/live2d_development', room=flask.request.sid)
        return

    state = session.renderer.configuration.states.get(state_name)
    if not state:
        socket.emit('error', {
            'message': f'Invalid state name: {state_name}'}, namespace='/live2d_development', room=flask.request.sid)

    composed_animation = session.renderer.compose_animation_group(
        state[animation_group_id])
    frames = session.renderer.render_animation_no_callback(composed_animation)
    rendered_mp4 = session.renderer.convert_composed_frames_to_mp4(frames)

    socket.emit('render_animation_group_result', rendered_mp4, namespace='/live2d_development', room=flask.request.sid)


@socket.on('disconnect', namespace='/live2d_development')
def on_disconnect():
    live2d_session_manager.shutdownSessionBySid(flask.session['auth'])
    pass


if __name__ == '__main__':
    logger.Logger.log(f'Starting THA4 API middleware server on port 3623')
    app.secret_key = 'THA4 API Middleware'
    socket.run(app, debug=False, host='0.0.0.0', port=3623)
