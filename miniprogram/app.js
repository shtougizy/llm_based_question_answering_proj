App({
  globalData: {
    token: '',
    user: null,
    baseUrl: 'https://home.dfdfh.top'
  },

  onLaunch() {
    const token = wx.getStorageSync('token');
    const user  = wx.getStorageSync('user');
    if (token && user) {
      this.globalData.token = token;
      this.globalData.user  = user;
    }
  },

  // 检查登录状态，未登录则跳转登录页
  checkLogin() {
    if (!this.globalData.token) {
      wx.navigateTo({ url: '/pages/login/login' });
      return false;
    }
    return true;
  },

  // 统一请求方法
  request(options) {
    const { url, method = 'GET', data, header = {}, success, fail } = options;
    const token = this.globalData.token;
    if (token) header['Authorization'] = 'Bearer ' + token;
    header['Content-Type'] = header['Content-Type'] || 'application/json';

    wx.request({
      url: this.globalData.baseUrl + url,
      method,
      data,
      header,
      success: (res) => {
        if (res.statusCode === 401) {
          // token 失效，清除登录态
          this.globalData.token = '';
          this.globalData.user  = null;
          wx.removeStorageSync('token');
          wx.removeStorageSync('user');
          wx.navigateTo({ url: '/pages/login/login' });
          return;
        }
        success && success(res);
      },
      fail: (err) => {
        wx.showToast({ title: '网络请求失败', icon: 'none' });
        fail && fail(err);
      }
    });
  },

  // 上传图片
  uploadFile(options) {
    const { filePath, success, fail } = options;
    const token = this.globalData.token;
    const username = this.globalData.user ? this.globalData.user.username : 'default';

    wx.uploadFile({
      url: this.globalData.baseUrl + '/api/search/image/async',
      filePath,
      name: 'file',
      formData: { username, need_visualization: 'false' },
      header: { 'Authorization': token ? 'Bearer ' + token : '' },
      success,
      fail: () => {
        wx.showToast({ title: '上传失败', icon: 'none' });
        fail && fail();
      }
    });
  }
});
