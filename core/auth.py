"""
认证模块：JWT、密码哈希、验证码
"""
import random
import string
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jose import JWTError, jwt
from passlib.context import CryptContext
import hashlib

logger = logging.getLogger(__name__)

# ==================== 配置 ====================
SECRET_KEY = "CHANGE_THIS_IN_PRODUCTION_USE_RANDOM_32_CHARS"  # 生产环境请改为随机字符串
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7   # 7天
GUEST_TOKEN_EXPIRE_MINUTES  = 60 * 24        # 游客 1天

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 内存中存储验证码 {phone: {"code": "123456", "expires": datetime}}
_sms_store: dict = {}

# ==================== 密码 ====================

def _prepare_password(password: str) -> str:
    """对密码做 SHA256，避免 bcrypt 72字节限制"""
    return hashlib.sha256(password.encode()).hexdigest()

def hash_password(password: str) -> str:
    return pwd_context.hash(_prepare_password(password))

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(_prepare_password(plain), hashed)


# ==================== JWT ====================

def create_access_token(
    user_id: int,
    username: str,
    role: str = "student",
    is_guest: bool = False,
    expires_minutes: int = None,
) -> str:
    if expires_minutes is None:
        expires_minutes = GUEST_TOKEN_EXPIRE_MINUTES if is_guest else ACCESS_TOKEN_EXPIRE_MINUTES

    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "is_guest": is_guest,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """解析 JWT，返回 payload 或 None"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ==================== 验证码 ====================

def generate_sms_code(phone: str) -> str:
    """生成6位验证码并存储（有效期5分钟）"""
    code = "".join(random.choices(string.digits, k=6))
    _sms_store[phone] = {
        "code": code,
        "expires": datetime.utcnow() + timedelta(minutes=5),
    }
    logger.info(f"[SMS] 手机 {phone} 验证码：{code}")  # 开发环境直接打印，生产环境接短信API
    return code


def verify_sms_code(phone: str, code: str) -> bool:
    """验证短信验证码"""
    record = _sms_store.get(phone)
    if not record:
        return False
    if datetime.utcnow() > record["expires"]:
        del _sms_store[phone]
        return False
    if record["code"] != code:
        return False
    del _sms_store[phone]  # 验证成功后删除
    return True


def send_sms(phone: str, code: str):
    """
    发送短信接口占位函数。
    生产环境接入阿里云/腾讯云短信SDK：
    
    from aliyunsdkcore.client import AcsClient
    from aliyunsdkcore.request import CommonRequest
    ...
    
    目前直接在日志里打印验证码（开发模式）。
    """
    logger.info(f"[SMS MOCK] 向 {phone} 发送验证码: {code}")
    # TODO: 接入真实短信服务
