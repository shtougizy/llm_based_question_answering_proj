const API = require('../../utils/api');
const app = getApp();

Page({
  data: {
    loading: false,
    loadingText: ''
  },

  onLoad() {
    // 已登录则直接返回
    if (app.globalData.token) {
      wx.navigateBack({ delta: 1 });
    }
  },

  // 微信一键登录
  async onWechatLogin() {
    this.setData({ loading: true, loadingText: '登录中...' });
    try {
      const { code } = await new Promise((resolve, reject) =>
        wx.login({ success: resolve, fail: reject })
      );
      const data = await API.wechatLogin(code);
      app.globalData.token = data.token;
      app.globalData.user  = data.user;
      wx.setStorageSync('token', data.token);
      wx.setStorageSync('user', data.user);
      wx.showToast({ title: data.is_new ? '注册成功！' : '登录成功！', icon: 'success' });
      setTimeout(() => {
        const pages = getCurrentPages();
        if (pages.length > 1) wx.navigateBack({ delta: 1 });
        else wx.switchTab({ url: '/pages/index/index' });
      }, 800);
    } catch(e) {
      wx.showToast({ title: '登录失败，请重试', icon: 'none' });
    } finally {
      this.setData({ loading: false });
    }
  },

  // 游客访问
  async onGuestLogin() {
    this.setData({ loading: true, loadingText: '进入中...' });
    try {
      const data = await API.guestLogin();
      app.globalData.token = data.token;
      app.globalData.user  = data.user;
      wx.setStorageSync('token', data.token);
      wx.setStorageSync('user', data.user);
      const pages = getCurrentPages();
      if (pages.length > 1) wx.navigateBack({ delta: 1 });
      else wx.switchTab({ url: '/pages/index/index' });
    } catch(e) {
      wx.showToast({ title: '进入失败', icon: 'none' });
    } finally {
      this.setData({ loading: false });
    }
  }
});
