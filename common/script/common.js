$(document).ready(function($){
    $(".title", ".toggle-box").addClass('close');
    $(".content", ".toggle-box").css('display', 'none');
    $(".title", ".toggle-box").click(function(){
        $(this).parent().children(".content").toggle("blind", "up", 200);
        $(this).toggleClass('open');
	$(this).toggleClass('close');
    });
});

