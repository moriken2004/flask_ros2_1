import rclpy
from rclpy.node import Node
from flask import Flask, render_template, request, jsonify, Response
import threading
import time
import sys
import traceback
import json
import base64
import cv2
import numpy as np
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import asdict

# ROS2メッセージ
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

# 人物検出システムをインポート
from paste import PersonDetectionSystem, SystemConfig, PersonDetection

class CameraManager:
    """カメラ管理クラス"""
    
    def __init__(self):
        self.available_cameras = {}
        self.current_camera_id = 0
        self.scan_lock = threading.Lock()
        
    def scan_cameras(self):
        """利用可能なカメラをスキャン"""
        with self.scan_lock:
            cameras = {}
            
            # カメラID 0-9 をテスト
            for camera_id in range(10):
                try:
                    cap = cv2.VideoCapture(camera_id)
                    if cap.isOpened():
                        # カメラの情報を取得
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        # テストフレームを取得して実際に動作するか確認
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            cameras[camera_id] = {
                                'id': camera_id,
                                'name': f'Camera {camera_id}',
                                'width': width or 640,
                                'height': height or 480,
                                'fps': fps or 30,
                                'status': 'available',
                                'device_path': f'/dev/video{camera_id}'
                            }
                            print(f"Camera {camera_id} detected: {width}x{height} @ {fps}fps")
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"Error testing camera {camera_id}: {e}")
                    continue
            
            self.available_cameras = cameras
            return cameras
    
    def get_available_cameras(self):
        """利用可能なカメラリストを取得"""
        return self.available_cameras.copy()
    
    def get_current_camera_info(self):
        """現在のカメラ情報を取得"""
        if self.current_camera_id in self.available_cameras:
            camera_info = self.available_cameras[self.current_camera_id].copy()
            camera_info['status'] = 'active'
            return camera_info
        return None

class PersonFollowingController:
    """人物追従制御クラス"""
    
    def __init__(self, ros2_interface):
        self.ros2_interface = ros2_interface
        self.following_enabled = True
        
        # 制御パラメータ
        self.linear_speed = 0.3   # 前後移動速度 [m/s]
        self.angular_speed = 0.5  # 旋回速度 [rad/s]
        
    def enable_following(self):
        """人物追従を有効化"""
        self.following_enabled = True
        
    def disable_following(self):
        """人物追従を無効化"""
        self.following_enabled = False
        self.ros2_interface.stop_robot()
        
    def process_detections(self, detections: List[PersonDetection]):
        """検出結果を処理してロボット制御コマンドを送信"""
        if not self.following_enabled or not detections:
            self.ros2_interface.stop_robot()
            return
        
        # 最初に検出された人物を追従対象とする
        target_person = detections[0]
        direction = target_person.direction
        
        # 移動方向に応じたロボット制御
        if direction == "moving_toward":
            # 人物が前に動いている → ロボットは後進
            self.ros2_interface.move_robot(-self.linear_speed, 0.0)
            print(f"Person moving toward camera -> Robot backing up")
            
        elif direction == "moving_away":
            # 人物が遠ざかっている → ロボットは前進
            self.ros2_interface.move_robot(self.linear_speed, 0.0)
            print(f"Person moving away -> Robot moving forward")
            
        elif direction == "moving_right":
            # 人物が右に動いている → ロボットは右旋回
            self.ros2_interface.move_robot(0.0, -self.angular_speed)
            print(f"Person moving right -> Robot turning right")
            
        elif direction == "moving_left":
            # 人物が左に動いている → ロボットは左旋回
            self.ros2_interface.move_robot(0.0, self.angular_speed)
            print(f"Person moving left -> Robot turning left")
            
        elif direction == "stationary" or direction == "unknown":
            # 人物が停止またはunknown → ロボットは停止
            self.ros2_interface.stop_robot()
            print(f"Person stationary/unknown -> Robot stopped")
            
    def set_speeds(self, linear_speed: float, angular_speed: float):
        """制御速度を設定"""
        self.linear_speed = abs(linear_speed)
        self.angular_speed = abs(angular_speed)
        
    def get_status(self):
        """追従制御の状態を取得"""
        return {
            'following_enabled': self.following_enabled,
            'linear_speed': self.linear_speed,
            'angular_speed': self.angular_speed
        }

class VisionSystemManager:
    """人物検出システム管理クラス"""
    
    def __init__(self, person_following_controller=None):
        # カメラマネージャー
        self.camera_manager = CameraManager()
        
        # 人物追従制御クラス
        self.person_following_controller = person_following_controller
        
        # 人物検出システムの設定
        self.vision_config = SystemConfig()
        self.vision_config.camera_id = 0
        self.vision_config.save_debug_images = False
        
        # 検出システム初期化
        self.vision_system = None
        self.vision_active = False
        self.vision_thread = None
        
        # 検出結果の保存
        self.latest_detections: List[PersonDetection] = []
        self.detection_history: List[Dict] = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 統計情報
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'fps': 0.0,
            'current_camera': None
        }
        
        # 初期化
        self._initialize_system()
    
    def _initialize_system(self):
        """システム初期化"""
        try:
            # カメラをスキャン
            cameras = self.camera_manager.scan_cameras()
            print(f"Found {len(cameras)} cameras")
            
            if cameras:
                # 最初のカメラで初期化
                first_camera_id = min(cameras.keys())
                self._create_vision_system(first_camera_id)
            else:
                print("No cameras found")
                
        except Exception as e:
            print(f"Vision system initialization failed: {e}")
    
    def _create_vision_system(self, camera_id):
        """指定されたカメラでビジョンシステムを作成"""
        try:
            # 既存のシステムを停止
            if self.vision_system and hasattr(self.vision_system, 'cap') and self.vision_system.cap:
                self.vision_system.cap.release()
            
            # 新しいシステムを作成
            self.vision_config.camera_id = camera_id
            self.vision_system = PersonDetectionSystem(self.vision_config)
            
            if self.vision_system and self.vision_system.cap and self.vision_system.cap.isOpened():
                self.camera_manager.current_camera_id = camera_id
                self.vision_active = True
                self.stats['current_camera'] = self.camera_manager.get_current_camera_info()
                print(f"Vision system created with camera {camera_id}")
                return True
            else:
                raise Exception(f"Failed to open camera {camera_id}")
                
        except Exception as e:
            print(f"Failed to create vision system with camera {camera_id}: {e}")
            self.vision_system = None
            self.vision_active = False
            return False
    
    def switch_camera(self, camera_id):
        """カメラを切り替え"""
        if camera_id not in self.camera_manager.available_cameras:
            return False, f"Camera {camera_id} not available"
        
        if camera_id == self.camera_manager.current_camera_id:
            return True, f"Camera {camera_id} is already active"
        
        try:
            # ビジョンループを停止
            old_active = self.vision_active
            self.vision_active = False
            
            # 少し待機
            time.sleep(0.5)
            
            # 新しいカメラでシステムを作成
            success = self._create_vision_system(camera_id)
            
            if success:
                # ビジョンループを再開
                if old_active:
                    self.vision_active = True
                    self.start_vision_thread()
                
                return True, f"Successfully switched to camera {camera_id}"
            else:
                return False, f"Failed to switch to camera {camera_id}"
                
        except Exception as e:
            return False, f"Error switching camera: {e}"
    
    def start_vision_thread(self):
        """人物検出スレッドを開始"""
        if not self.vision_active or not self.vision_system:
            return False
        
        # 既存のスレッドが動作中の場合は停止
        if self.vision_thread and self.vision_thread.is_alive():
            return True
        
        self.vision_thread = threading.Thread(
            target=self._vision_loop,
            daemon=True
        )
        self.vision_thread.start()
        return True
    
    def _vision_loop(self):
        """人物検出ループ"""
        if not self.vision_system or not self.vision_system.cap:
            return
        
        fps_counter = 0
        last_fps_time = time.time()
        
        try:
            while self.vision_active and self.vision_system and self.vision_system.cap:
                ret, frame = self.vision_system.cap.read()
                if not ret:
                    print("Failed to read frame")
                    continue
                
                # 人物検出実行
                detections = self.vision_system.detect_persons(frame)
                
                # 人物追従制御を実行
                if self.person_following_controller:
                    self.person_following_controller.process_detections(detections)
                
                # 結果を描画
                annotated_frame = self.vision_system.draw_detections(frame, detections)
                
                # 結果を保存
                with self.frame_lock:
                    self.latest_detections = detections
                    self.latest_frame = annotated_frame
                    
                    # 統計更新
                    self.stats['total_frames'] += 1
                    self.stats['total_detections'] += len(detections)
                    self.stats['current_camera'] = self.camera_manager.get_current_camera_info()
                    
                    # 検出履歴に追加
                    if detections:
                        detection_record = {
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': self.stats['total_frames'],
                            'camera_id': self.camera_manager.current_camera_id,
                            'detections': [
                                {
                                    'person_id': d.person_id,
                                    'direction': d.direction,
                                    'confidence': d.confidence,
                                    'center': d.center
                                }
                                for d in detections
                            ]
                        }
                        self.detection_history.append(detection_record)
                        
                        # 履歴サイズ制限
                        if len(self.detection_history) > 100:
                            self.detection_history.pop(0)
                
                # FPS計算
                fps_counter += 1
                if fps_counter % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - last_fps_time
                    if elapsed > 0:
                        self.stats['fps'] = 30 / elapsed
                    last_fps_time = current_time
                
                time.sleep(0.03)  # 約30FPS
                
        except Exception as e:
            print(f"Vision loop error: {e}")
        finally:
            print("Vision loop ended")
    
    def get_latest_detections(self):
        """最新の検出結果を取得"""
        with self.frame_lock:
            return self.latest_detections.copy()
    
    def get_detection_history(self):
        """検出履歴を取得"""
        with self.frame_lock:
            return self.detection_history.copy()
    
    def get_latest_frame(self):
        """最新のフレームを取得"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def get_stats(self):
        """統計情報を取得"""
        current_stats = self.stats.copy()
        current_stats['uptime'] = time.time() - self.stats['start_time']
        return current_stats
    
    def rescan_cameras(self):
        """カメラを再スキャン"""
        return self.camera_manager.scan_cameras()
    
    def get_available_cameras(self):
        """利用可能なカメラを取得"""
        return self.camera_manager.get_available_cameras()
    
    def get_current_camera(self):
        """現在のカメラ情報を取得"""
        return self.camera_manager.get_current_camera_info()

class ROS2Interface(Node):
    """ROS2との通信を担当するクラス"""
    
    def __init__(self):
        super().__init__('flask_robot_controller')
        
        # Publisher設定
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.command_pub = self.create_publisher(String, '/robot_command', 10)
        
        # Subscriber設定
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        
        # ロボットの状態を保存
        self.robot_status = {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'velocity': {'linear': 0, 'angular': 0},
            'laser_data': {'min_distance': 0, 'ranges_count': 0},
            'last_update': time.time()
        }
        
    def laser_callback(self, msg):
        """レーザーセンサーデータのコールバック"""
        try:
            if msg.ranges:
                valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
                if valid_ranges:
                    min_distance = min(valid_ranges)
                    self.robot_status['laser_data'] = {
                        'min_distance': min_distance,
                        'ranges_count': len(msg.ranges)
                    }
                    self.robot_status['last_update'] = time.time()
        except Exception as e:
            self.get_logger().error(f'Laser callback error: {e}')
    
    def move_robot(self, linear_x, angular_z):
        """ロボットを移動させる"""
        try:
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_vel_pub.publish(twist)
            
            # 状態更新
            self.robot_status['velocity']['linear'] = linear_x
            self.robot_status['velocity']['angular'] = angular_z
            
            return True
        except Exception as e:
            self.get_logger().error(f'Move robot error: {e}')
            return False
    
    def stop_robot(self):
        """ロボットを停止"""
        return self.move_robot(0.0, 0.0)
    
    def send_custom_command(self, command):
        """カスタムコマンドを送信"""
        try:
            msg = String()
            msg.data = command
            self.command_pub.publish(msg)
            self.get_logger().info(f'Command sent: {command}')
            return True
        except Exception as e:
            self.get_logger().error(f'Send command error: {e}')
            return False
    
    def get_robot_status(self):
        """ロボットの状態を取得"""
        return self.robot_status.copy()

# Flaskアプリケーションの作成
app = Flask(__name__)
app.config['DEBUG'] = True

# グローバル変数
ros2_interface = None
vision_manager = None
person_following_controller = None

@app.errorhandler(Exception)
def handle_exception(e):
    """例外ハンドラー"""
    print(f"Error: {e}")
    print(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': str(e)
    }), 500

# ロボット制御ページ
@app.route('/')
def index():
    """メインページ"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template error: {e}")
        return render_template('fallback.html')

# 人物検出ページ
@app.route('/vision')
def vision_page():
    """人物検出ページ"""
    try:
        return render_template('vision.html')
    except Exception as e:
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Vision System</title></head>
        <body>
            <h1>Vision System Error</h1>
            <p>Template not found. Please create templates/vision.html</p>
        </body>
        </html>
        """

# ロボット制御API
@app.route('/api/status', methods=['GET'])
def get_status():
    """ロボットの状態を取得"""
    try:
        if ros2_interface is None:
            return jsonify({'success': False, 'error': 'ROS2 interface not initialized'}), 500
        
        status = ros2_interface.get_robot_status()
        
        # 人物追従制御の状態も追加
        if person_following_controller:
            status['person_following'] = person_following_controller.get_status()
        
        return jsonify({'success': True, 'data': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/move', methods=['POST'])
def move_robot():
    """ロボット移動コマンド"""
    try:
        if ros2_interface is None:
            return jsonify({'success': False, 'error': 'ROS2 interface not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        linear_x = data.get('linear_x', 0)
        angular_z = data.get('angular_z', 0)
        success = ros2_interface.move_robot(linear_x, angular_z)
        
        return jsonify({
            'success': success,
            'message': f'Robot moved: linear={linear_x}, angular={angular_z}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_robot():
    """ロボット停止"""
    try:
        if ros2_interface is None:
            return jsonify({'success': False, 'error': 'ROS2 interface not initialized'}), 500
        
        success = ros2_interface.stop_robot()
        return jsonify({'success': success, 'message': 'Robot stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/command', methods=['POST'])
def send_command():
    """カスタムコマンド送信"""
    try:
        if ros2_interface is None:
            return jsonify({'success': False, 'error': 'ROS2 interface not initialized'}), 500
        
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'success': False, 'error': 'Command is required'}), 400
        
        command = data['command']
        success = ros2_interface.send_custom_command(command)
        
        return jsonify({'success': success, 'message': f'Command sent: {command}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 人物追従制御API
@app.route('/api/following/enable', methods=['POST'])
def enable_following():
    """人物追従を有効化"""
    try:
        if person_following_controller is None:
            return jsonify({'success': False, 'error': 'Person following controller not initialized'}), 500
        
        person_following_controller.enable_following()
        return jsonify({'success': True, 'message': 'Person following enabled'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/following/disable', methods=['POST'])
def disable_following():
    """人物追従を無効化"""
    try:
        if person_following_controller is None:
            return jsonify({'success': False, 'error': 'Person following controller not initialized'}), 500
        
        person_following_controller.disable_following()
        return jsonify({'success': True, 'message': 'Person following disabled'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/following/set_speeds', methods=['POST'])
def set_following_speeds():
    """人物追従の速度設定"""
    try:
        if person_following_controller is None:
            return jsonify({'success': False, 'error': 'Person following controller not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        linear_speed = data.get('linear_speed', 0.3)
        angular_speed = data.get('angular_speed', 0.5)
        
        person_following_controller.set_speeds(linear_speed, angular_speed)
        
        return jsonify({
            'success': True,
            'message': f'Speeds set: linear={linear_speed}, angular={angular_speed}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 人物検出API
@app.route('/api/vision/status', methods=['GET'])
def get_vision_status():
    """人物検出システムの状態を取得"""
    try:
        if vision_manager is None:
            return jsonify({'success': False, 'error': 'Vision system not initialized'}), 500
        
        stats = vision_manager.get_stats()
        detections = vision_manager.get_latest_detections()
        
        return jsonify({
            'success': True,
            'active': vision_manager.vision_active,
            'stats': stats,
            'current_detections': len(detections),
            'detections': [
                {
                    'person_id': d.person_id,
                    'direction': d.direction,
                    'confidence': d.confidence,
                    'center': d.center
                }
                for d in detections
            ]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vision/history', methods=['GET'])
def get_vision_history():
    """検出履歴を取得"""
    try:
        if vision_manager is None:
            return jsonify({'success': False, 'error': 'Vision system not initialized'}), 500
        
        history = vision_manager.get_detection_history()
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vision/cameras', methods=['GET'])
def get_cameras():
    """利用可能なカメラ一覧を取得"""
    try:
        if vision_manager is None:
            return jsonify({'success': False, 'error': 'Vision system not initialized'}), 500
        
        cameras = vision_manager.get_available_cameras()
        current_camera = vision_manager.get_current_camera()
        
        return jsonify({
            'success': True,
            'cameras': cameras,
            'current_camera': current_camera
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vision/rescan_cameras', methods=['POST'])
def rescan_cameras():
    """カメラを再スキャン"""
    try:
        if vision_manager is None:
            return jsonify({'success': False, 'error': 'Vision system not initialized'}), 500
        
        cameras = vision_manager.rescan_cameras()
        
        return jsonify({
            'success': True,
            'cameras': cameras,
            'message': f'Found {len(cameras)} cameras'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vision/switch_camera', methods=['POST'])
def switch_camera():
    """カメラを切り替え"""
    try:
        if vision_manager is None:
            return jsonify({'success': False, 'error': 'Vision system not initialized'}), 500
        
        data = request.get_json()
        if not data or 'camera_id' not in data:
            return jsonify({'success': False, 'error': 'camera_id is required'}), 400
        
        camera_id = int(data['camera_id'])
        success, message = vision_manager.switch_camera(camera_id)
        
        if success:
            current_camera = vision_manager.get_current_camera()
            return jsonify({
                'success': True,
                'message': message,
                'current_camera': current_camera
            })
        else:
            return jsonify({'success': False, 'error': message}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vision/video_feed')
def video_feed():
    """ビデオストリーミング"""
    def generate():
        while True:
            if vision_manager is None:
                break
            
            frame = vision_manager.get_latest_frame()
            if frame is not None:
                # JPEGエンコード
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.05)  # 20FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def ros2_spin_thread(interface):
    """ROS2のスピン処理を別スレッドで実行"""
    try:
        rclpy.spin(interface)
    except Exception as e:
        print(f"ROS2 spin error: {e}")

def main():
    """メイン関数"""
    global ros2_interface, vision_manager, person_following_controller
    
    try:
        print("Flask ROS2 Robot Controller with Vision and Person Following starting...")
        
        # ROS2初期化
        rclpy.init()
        ros2_interface = ROS2Interface()
        
        # 人物追従制御クラス初期化
        person_following_controller = PersonFollowingController(ros2_interface)
        
        # ROS2スピン用のスレッド開始
        ros2_thread = threading.Thread(
            target=ros2_spin_thread,
            args=(ros2_interface,),
            daemon=True
        )
        ros2_thread.start()
        
        # 人物検出システム初期化（人物追従制御クラスを渡す）
        vision_manager = VisionSystemManager(person_following_controller)
        if vision_manager.start_vision_thread():
            print("Vision system started successfully")
        else:
            print("Vision system failed to start (camera may not be available)")
        
        print("Access: http://localhost:5000 (Robot Control)")
        print("Access: http://localhost:5000/vision (Person Detection)")
        print("\nAvailable person following APIs:")
        print("  POST /api/following/enable - Enable person following")
        print("  POST /api/following/disable - Disable person following")
        print("  POST /api/following/set_speeds - Set following speeds")
        print("\nAvailable camera switch APIs:")
        print("  GET  /api/vision/cameras - Get available cameras")
        print("  POST /api/vision/rescan_cameras - Rescan cameras")
        print("  POST /api/vision/switch_camera - Switch camera")
        
        # Flaskアプリ実行
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if person_following_controller:
            person_following_controller.disable_following()
        if vision_manager:
            vision_manager.vision_active = False
        if ros2_interface is not None:
            ros2_interface.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
