var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
    legend: {
        data: {{legends}}
    },
    xAxis: {
      type: 'category',
      data: {{labels}},
      axisTick: { show: false }
    },
    yAxis: {
        name: 'Return',
        nameLocation: 'middle',
        nameGap: '50',
        type: 'value',
        axisLabel: {
            formatter: '{value} %'
        },
    },
    series: [
        {% for data in datasets %}
        {
            name: {{data["name"]}},
            type: 'bar',
            data: {{data["data"]}}
        },
        {% endfor %}
    ],
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    }
});

window.addEventListener('resize',function(){
  {{chart_id}}.resize();
})