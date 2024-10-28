using Microsoft.EntityFrameworkCore;
using System.IdentityModel.Tokens.Jwt;
using VertoTest.Context;
using VertoTest.Models;
using VertoTest.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllersWithViews();
builder.Services.AddDbContext<CalendarContext>(options => options.UseSqlServer(builder.Configuration.GetConnectionString("VertoDbConnection")));
builder.Services.AddScoped<EventInterface, EventService>();
builder.Services.AddMemoryCache();

var jwtSettings = builder.Configuration.GetSection("Jwt");
builder.Services.Configure<JwtSettings>(builder.Configuration.GetSection("Jwt"));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=HomePage}/{id?}");

app.Run();
