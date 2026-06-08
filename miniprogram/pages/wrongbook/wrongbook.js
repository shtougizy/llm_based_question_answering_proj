const API = require('../../utils/api');

Page({
  data: {
    tab: 0,
    wrongs: [],
    loading: true,
    clusters: [],
    clusterLoading: false,
    report: '',
    reportLoading: false,
    practicePlan: [],
    practiceLoading: false,
    expandedId: null,
    stats: { total: 0, subjects: 0, knowledges: 0 },
    errorMsg: ''
  },

  onShow() { this.loadWrongs(); },

  switchTab(e) {
    this.setData({ tab: +e.currentTarget.dataset.tab, errorMsg: '' });
  },

  async loadWrongs() {
    this.setData({ loading: true });
    try {
      const data = await API.getWrongBook();
      const wrongs = data.wrong_questions || [];
      const subjects = new Set(wrongs.map(q => q.subject).filter(Boolean)).size;
      const kns = new Set(wrongs.flatMap(q => q.knowledges || [])).size;
      this.setData({ wrongs, stats: { total: wrongs.length, subjects, knowledges: kns } });
    } catch(e) {
      wx.showToast({ title: '加载失败: ' + (e.message || ''), icon: 'none', duration: 3000 });
    } finally {
      this.setData({ loading: false });
    }
  },

  toggleExpand(e) {
    const id = e.currentTarget.dataset.id;
    this.setData({ expandedId: this.data.expandedId === id ? null : id });
  },

  loadClusters() {
    const self = this;
    self.setData({ clusterLoading: true, clusters: [], errorMsg: '' });
    const app = getApp();
    wx.request({
      url: app.globalData.baseUrl + '/api/cluster-analysis?username=' +
        (app.globalData.user ? app.globalData.user.username : 'default'),
      header: { 'Authorization': app.globalData.token ? 'Bearer ' + app.globalData.token : '' },
      timeout: 90000,
      success(res) {
        if (res.statusCode === 200) {
          self.setData({ clusters: res.data.clusters || [] });
          if ((res.data.clusters || []).length === 0) {
            self.setData({ errorMsg: '暂无足够错题进行聚类分析' });
          }
        } else {
          self.setData({ errorMsg: '接口返回错误: ' + res.statusCode });
        }
      },
      fail(err) {
        self.setData({ errorMsg: '网络请求失败: ' + (err.errMsg || '') });
        wx.showToast({ title: '分析失败，请检查网络', icon: 'none', duration: 3000 });
      },
      complete() {
        self.setData({ clusterLoading: false });
      }
    });
  },

  loadReport() {
    const self = this;
    self.setData({ reportLoading: true, report: '', errorMsg: '' });
    const app = getApp();
    wx.request({
      url: app.globalData.baseUrl + '/api/wrong-report?username=' +
        (app.globalData.user ? app.globalData.user.username : 'default'),
      header: { 'Authorization': app.globalData.token ? 'Bearer ' + app.globalData.token : '' },
      timeout: 120000,
      success(res) {
        if (res.statusCode === 200) {
          let report = res.data.report;
          if (Array.isArray(report)) report = report.filter(s => s).join('\n\n');
          self.setData({ report: report || '暂无内容' });
        } else {
          self.setData({ errorMsg: '生成失败: ' + res.statusCode });
        }
      },
      fail(err) {
        self.setData({ errorMsg: '网络超时，报告生成需要较长时间，请稍后重试' });
        wx.showToast({ title: '生成失败，请重试', icon: 'none', duration: 3000 });
      },
      complete() {
        self.setData({ reportLoading: false });
      }
    });
  },

  loadPractice() {
    const self = this;
    self.setData({ practiceLoading: true, practicePlan: [], errorMsg: '' });
    const app = getApp();
    wx.request({
      url: app.globalData.baseUrl + '/api/practice-plan?username=' +
        (app.globalData.user ? app.globalData.user.username : 'default') +
        '&questions_per_cluster=3',
      header: { 'Authorization': app.globalData.token ? 'Bearer ' + app.globalData.token : '' },
      timeout: 120000,
      success(res) {
        if (res.statusCode === 200) {
          self.setData({ practicePlan: res.data.plan || [] });
          if ((res.data.plan || []).length === 0) {
            self.setData({ errorMsg: '暂无练习题，请先标记一些错题' });
          }
        } else {
          self.setData({ errorMsg: '生成失败: ' + res.statusCode });
        }
      },
      fail(err) {
        self.setData({ errorMsg: '网络超时，练习题生成需要较长时间，请稍后重试' });
        wx.showToast({ title: '生成失败，请重试', icon: 'none', duration: 3000 });
      },
      complete() {
        self.setData({ practiceLoading: false });
      }
    });
  },

  toggleAnswer(e) {
    const pi = e.currentTarget.dataset.pi;
    const qi = e.currentTarget.dataset.qi;
    const shown = e.currentTarget.dataset.shown;
    const key = 'practicePlan[' + pi + '].practice_questions[' + qi + '].showAnswer';
    this.setData({ [key]: !shown });
  }
});
