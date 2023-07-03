var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
    grid: {
		top: '5%',
		height: '60%'
	},
    xAxis: {
        type: 'category',
        data: {{labels}},
        axisTick: {
          alignWithLabel: true
        },
        axisLabel: {
            rotate: 60,
			interval: 0,
			textStyle: { fontSize:11 }
        },
    },
    yAxis: {
        type: 'value'
    },
    series: [
        {
            type: 'bar',
            data: {{data}}
        }
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