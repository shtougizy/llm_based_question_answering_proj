const API = require('../../utils/api');

Page({
  data: {
    records: [],
    filtered: [],
    loading: true,
    stats: { total: 0, wrong: 0, correct: 0, rate: '-' },
    filterStatus: 'all',
    keyword: '',
    expandedId: null
  },

  onShow() {
    this.loadHistory();
  },

  async loadHistory() {
    this.setData({ loading: true });
    try {
      const data = await API.getHistory(200);
      const records = data.records || [];
      this.setData({ records });
      this.applyFilter();
      this.calcStats(records);
    } catch(e) {
      wx.showToast({ title: '加载失败', icon: 'none' });
    } finally {
      this.setData({ loading: false });
    }
  },

  calcStats(records) {
    const total = records.length;
    const wrong = records.filter(r => r.is_wrong).length;
    const correct = total - wrong;
    const rate = total > 0 ? Math.round(correct / total * 100) + '%' : '-';
    this.setData({ stats: { total, wrong, correct, rate } });
  },

  onFilterChange(e) {
    this.setData({ filterStatus: e.currentTarget.dataset.status });
    this.applyFilter();
  },

  onKeywordInput(e) {
    this.setData({ keyword: e.detail.value });
    this.applyFilter();
  },

  applyFilter() {
    const { records, filterStatus, keyword } = this.data;
    let filtered = records.filter(r => {
      if (filterStatus === 'wrong' && !r.is_wrong) return false;
      if (filterStatus === 'correct' && r.is_wrong) return false;
      if (keyword && !r.question_text.includes(keyword)) return false;
      return true;
    });
    this.setData({ filtered });
  },

  toggleExpand(e) {
    const id = e.currentTarget.dataset.id;
    this.setData({ expandedId: this.data.expandedId === id ? null : id });
  },

  async onMarkWrong(e) {
    const id = e.currentTarget.dataset.id;
    try {
      await API.markWrong(id);
      const records = this.data.records.map(r => r.id === id ? { ...r, is_wrong: true } : r);
      this.setData({ records });
      this.applyFilter();
      this.calcStats(records);
      wx.showToast({ title: '已加入错题本', icon: 'success' });
    } catch(e) {
      wx.showToast({ title: '操作失败', icon: 'none' });
    }
  }
});
