const app = getApp();

const API = {
  wechatLogin(code) {
    return new Promise((resolve, reject) => {
      app.request({
        url: `/api/auth/wechat-login?code=${code}`,
        method: 'POST',
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  guestLogin() {
    return new Promise((resolve, reject) => {
      app.request({
        url: '/api/auth/guest',
        method: 'POST',
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  searchText(questionText, needViz = false) {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: '/api/search/text',
        method: 'POST',
        data: { question_text: questionText, username, need_visualization: needViz },
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  markWrong(recordId) {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: '/api/mark-wrong',
        method: 'POST',
        data: { record_id: recordId, username },
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  getHistory(limit = 50) {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: `/api/history?username=${username}&limit=${limit}`,
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  getWrongBook() {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: `/api/wrong-book?username=${username}`,
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  getClusterAnalysis() {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: `/api/cluster-analysis?username=${username}`,
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  getWrongReport() {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: `/api/wrong-report?username=${username}`,
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  },

  getPracticePlan(n = 3) {
    const username = app.globalData.user ? app.globalData.user.username : 'default';
    return new Promise((resolve, reject) => {
      app.request({
        url: `/api/practice-plan?username=${username}&questions_per_cluster=${n}`,
        success: (res) => res.statusCode === 200 ? resolve(res.data) : reject(res.data),
        fail: reject
      });
    });
  }
};

module.exports = API;
